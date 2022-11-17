/**
 * Squared Euclidean distance functor, optimized version - filter by first n dims
 */
template <class T, int d>
struct LimitedL2 {
  typedef bool is_kdtree_distance;

  typedef T ElementType;
  typedef typename Accumulator<T>::Type ResultType;

  /**
   *  Compute the squared Euclidean distance between two vectors.
   *
   *	This is highly optimised, with loop unrolling, as it is one
   *	of the most expensive inner loops.
   *
   *	The computation of squared root at the end is omitted for
   *	efficiency.
   */
  template <typename Iterator1, typename Iterator2>
  ResultType operator()(Iterator1 a, Iterator2 b, size_t size, ResultType worst_dist = -1) const {
    ResultType result = ResultType();
    ResultType diff0, diff1, diff2, diff3;
    Iterator1 last = a + size;
    Iterator1 lastgroup = last - 3;

    /* Process 4 items with each loop for efficiency. */
    while (a < lastgroup) {
      diff0 = (ResultType)(a[0] - b[0]);
      diff1 = (ResultType)(a[1] - b[1]);
      diff2 = (ResultType)(a[2] - b[2]);
      diff3 = (ResultType)(a[3] - b[3]);
      result += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
      a += 4;
      b += 4;

      if ((worst_dist > 0) && (result > worst_dist)) {
        return result;
      }
    }
    /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
    while (a < last) {
      diff0 = (ResultType)(*a++ - *b++);
      result += diff0 * diff0;
    }
    return result;
  }

  /**
   *	Partial euclidean distance, using just one dimension. This is used by the
   *	kd-tree when computing partial distances while traversing the tree.
   *
   *	Squared root is omitted for efficiency.
   */
  template <typename U, typename V>
  inline ResultType accum_dist(const U& a, const V& b, int) const {
    return (a - b) * (a - b);
  }
};
