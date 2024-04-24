use std::borrow::Cow;
use std::fs::create_dir_all;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use atomic_refcell::AtomicRefCell;
use bitvec::prelude::BitSlice;
use common::types::PointOffsetType;

use crate::common::operation_error::{check_process_stopped, OperationResult};
use crate::common::Flusher;
use crate::data_types::named_vectors::CowVector;
use crate::data_types::primitive::PrimitiveVectorElement;
use crate::data_types::vectors::{
    TypedMultiDenseVector, TypedMultiDenseVectorRef, VectorElementType, VectorRef,
};
use crate::types::{Distance, MultiVectorConfig, VectorStorageDatatype};
use crate::vector_storage::chunked_mmap_vectors::ChunkedMmapVectors;
use crate::vector_storage::dense::dynamic_mmap_flags::DynamicMmapFlags;
use crate::vector_storage::{MultiVectorStorage, VectorStorage, VectorStorageEnum};

const VECTORS_DIR_PATH: &str = "vectors";
const OFFSETS_DIR_PATH: &str = "offsets";
const DELETED_DIR_PATH: &str = "deleted";

#[derive(Clone, Copy, Debug, PartialEq)]
struct MultivectorMmapOffset {
    offset: PointOffsetType,
    count: PointOffsetType,
}

pub struct AppendableMmapMultiDenseVectorStorage<T: PrimitiveVectorElement + 'static> {
    vectors: ChunkedMmapVectors<T>,
    offsets: ChunkedMmapVectors<MultivectorMmapOffset>,
    deleted: DynamicMmapFlags,
    distance: Distance,
    multi_vector_config: MultiVectorConfig,
    deleted_count: usize,
}

pub fn open_appendable_memmap_multi_vector_storage(
    path: &Path,
    dim: usize,
    distance: Distance,
    multi_vector_config: MultiVectorConfig,
    stopped: &AtomicBool,
) -> OperationResult<Arc<AtomicRefCell<VectorStorageEnum>>> {
    let storage = open_appendable_memmap_multi_vector_storage_impl::<VectorElementType>(
        path,
        dim,
        distance,
        multi_vector_config,
        stopped,
    )?;

    Ok(Arc::new(AtomicRefCell::new(
        VectorStorageEnum::MultiDenseAppendableMemmap(Box::new(storage)),
    )))
}

pub fn open_appendable_memmap_multi_vector_storage_impl<T: PrimitiveVectorElement>(
    path: &Path,
    dim: usize,
    distance: Distance,
    multi_vector_config: MultiVectorConfig,
    stopped: &AtomicBool,
) -> OperationResult<AppendableMmapMultiDenseVectorStorage<T>> {
    create_dir_all(path)?;

    let vectors_path = path.join(VECTORS_DIR_PATH);
    let offsets_path = path.join(OFFSETS_DIR_PATH);
    let deleted_path = path.join(DELETED_DIR_PATH);

    let vectors = ChunkedMmapVectors::open(&vectors_path, dim)?;
    let offsets = ChunkedMmapVectors::open(&offsets_path, 1)?;

    let num_vectors = vectors.len();

    let deleted: DynamicMmapFlags = DynamicMmapFlags::open(&deleted_path)?;

    let mut deleted_count = 0;

    for i in 0..num_vectors {
        if deleted.get(i) {
            deleted_count += 1;
        }
        check_process_stopped(stopped)?;
    }

    Ok(AppendableMmapMultiDenseVectorStorage {
        vectors,
        offsets,
        deleted,
        distance,
        multi_vector_config,
        deleted_count,
    })
}

impl<T: PrimitiveVectorElement + 'static> AppendableMmapMultiDenseVectorStorage<T> {
    /// Set deleted flag for given key. Returns previous deleted state.
    #[inline]
    fn set_deleted(&mut self, key: PointOffsetType, deleted: bool) -> OperationResult<bool> {
        if self.vectors.len() <= key as usize {
            return Ok(false);
        }

        if self.deleted.len() <= key as usize {
            self.deleted.set_len(key as usize + 1)?;
        }
        let previous = self.deleted.set(key, deleted);
        if !previous && deleted {
            self.deleted_count += 1;
        } else if previous && !deleted {
            self.deleted_count -= 1;
        }
        Ok(previous)
    }
}

impl<T: PrimitiveVectorElement> MultiVectorStorage<T> for AppendableMmapMultiDenseVectorStorage<T> {
    fn get_multi(&self, key: PointOffsetType) -> TypedMultiDenseVectorRef<T> {
        let mmap_offset = self.offsets.get(key as usize).unwrap().first().unwrap();
        let inner_vector = self
            .vectors
            .get_many(mmap_offset.offset, mmap_offset.count as usize)
            .expect("vector not found");
        TypedMultiDenseVectorRef {
            inner_vector,
            dim: self.vectors.dim(),
        }
    }

    fn multi_vector_config(&self) -> &MultiVectorConfig {
        &self.multi_vector_config
    }
}

impl<T: PrimitiveVectorElement> VectorStorage for AppendableMmapMultiDenseVectorStorage<T> {
    fn vector_dim(&self) -> usize {
        self.vectors.dim()
    }

    fn distance(&self) -> Distance {
        self.distance
    }

    fn datatype(&self) -> VectorStorageDatatype {
        T::datatype()
    }

    fn is_on_disk(&self) -> bool {
        true
    }

    fn total_vector_count(&self) -> usize {
        self.vectors.len()
    }

    fn get_vector(&self, key: PointOffsetType) -> CowVector {
        // TODO(colbert) borrow instead of clone
        let multivector = self.get_multi(key);
        let multivector = TypedMultiDenseVector {
            inner_vector: multivector.inner_vector.to_vec(),
            dim: multivector.dim,
        };
        CowVector::MultiDense(T::into_float_multivector(Cow::Owned(multivector)))
    }

    fn insert_vector(&mut self, key: PointOffsetType, vector: VectorRef) -> OperationResult<()> {
        let vector: &[VectorElementType] = vector.try_into()?;
        let vector = T::slice_from_float_cow(Cow::from(vector));
        self.vectors.insert(key, vector.as_ref())?;
        self.set_deleted(key, false)?;
        Ok(())
    }

    fn update_from(
        &mut self,
        other: &VectorStorageEnum,
        other_ids: &mut impl Iterator<Item = PointOffsetType>,
        stopped: &AtomicBool,
    ) -> OperationResult<Range<PointOffsetType>> {
        let start_index = self.vectors.len() as PointOffsetType;
        for point_id in other_ids {
            check_process_stopped(stopped)?;
            // Do not perform preprocessing - vectors should be already processed
            let other_deleted = other.is_deleted_vector(point_id);
            let other_vector = other.get_vector(point_id);
            let other_vector = T::slice_from_float_cow(Cow::try_from(other_vector)?);
            let new_id = self.vectors.push(other_vector.as_ref())?;
            self.set_deleted(new_id, other_deleted)?;
        }
        let end_index = self.vectors.len() as PointOffsetType;
        Ok(start_index..end_index)
    }

    fn flusher(&self) -> Flusher {
        Box::new({
            let vectors_flusher = self.vectors.flusher();
            let offsets_flusher = self.offsets.flusher();
            let deleted_flusher = self.deleted.flusher();
            move || {
                vectors_flusher()?;
                offsets_flusher()?;
                deleted_flusher()?;
                Ok(())
            }
        })
    }

    fn files(&self) -> Vec<PathBuf> {
        let mut files = self.vectors.files();
        files.extend(self.offsets.files());
        files.extend(self.deleted.files());
        files
    }

    fn delete_vector(&mut self, key: PointOffsetType) -> OperationResult<bool> {
        self.set_deleted(key, true)
    }

    fn is_deleted_vector(&self, key: PointOffsetType) -> bool {
        self.deleted.get(key)
    }

    fn deleted_vector_count(&self) -> usize {
        self.deleted_count
    }

    fn deleted_vector_bitslice(&self) -> &BitSlice {
        self.deleted.get_bitslice()
    }
}
