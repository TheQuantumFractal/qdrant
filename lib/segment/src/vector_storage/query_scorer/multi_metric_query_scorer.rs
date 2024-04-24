use std::marker::PhantomData;

use common::types::{PointOffsetType, ScoreType};

use super::score_multi;
use crate::data_types::vectors::{
    DenseVector, MultiDenseVector, TypedMultiDenseVectorRef, VectorElementType,
};
use crate::spaces::metric::Metric;
use crate::vector_storage::query_scorer::QueryScorer;
use crate::vector_storage::MultiVectorStorage;

pub struct MultiMetricQueryScorer<
    'a,
    TMetric: Metric<VectorElementType>,
    TVectorStorage: MultiVectorStorage<VectorElementType>,
> {
    vector_storage: &'a TVectorStorage,
    query: MultiDenseVector,
    metric: PhantomData<TMetric>,
}

impl<
        'a,
        TMetric: Metric<VectorElementType>,
        TVectorStorage: MultiVectorStorage<VectorElementType>,
    > MultiMetricQueryScorer<'a, TMetric, TVectorStorage>
{
    pub fn new(query: MultiDenseVector, vector_storage: &'a TVectorStorage) -> Self {
        let slices = query.multi_vectors();
        let preprocessed: DenseVector = slices
            .into_iter()
            .flat_map(|slice| TMetric::preprocess(slice.to_vec()))
            .collect();
        Self {
            query: MultiDenseVector::new(preprocessed, query.dim),
            vector_storage,
            metric: PhantomData,
        }
    }

    fn score_multi(
        &self,
        multi_dense_a: TypedMultiDenseVectorRef<VectorElementType>,
        multi_dense_b: TypedMultiDenseVectorRef<VectorElementType>,
    ) -> ScoreType {
        score_multi::<VectorElementType, TMetric>(
            self.vector_storage.multi_vector_config(),
            multi_dense_a,
            multi_dense_b,
        )
    }
}

impl<
        'a,
        TMetric: Metric<VectorElementType>,
        TVectorStorage: MultiVectorStorage<VectorElementType>,
    > QueryScorer<MultiDenseVector> for MultiMetricQueryScorer<'a, TMetric, TVectorStorage>
{
    #[inline]
    fn score_stored(&self, idx: PointOffsetType) -> ScoreType {
        self.score_multi(
            TypedMultiDenseVectorRef::from(&self.query),
            self.vector_storage.get_multi(idx),
        )
    }

    #[inline]
    fn score(&self, v2: &MultiDenseVector) -> ScoreType {
        self.score_multi(
            TypedMultiDenseVectorRef::from(&self.query),
            TypedMultiDenseVectorRef::from(v2),
        )
    }

    fn score_internal(&self, point_a: PointOffsetType, point_b: PointOffsetType) -> ScoreType {
        let v1 = self.vector_storage.get_multi(point_a);
        let v2 = self.vector_storage.get_multi(point_b);
        self.score_multi(v1, v2)
    }
}
