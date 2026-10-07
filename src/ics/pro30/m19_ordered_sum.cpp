// Port only np.add.at's row-ordered accumulation. Products and all decisions
// remain the source NumPy operations. No fast-math or fused product/addition.
#include <cstdint>
#include <cstring>
extern "C" int m19_ordered_sum(std::int64_t n,std::int64_t d,std::int64_t k,
    const std::int64_t* labels,const double* weighted,double* sums) {
    if(n<0||d<1||k<1)return 1;
    std::memset(sums,0,static_cast<std::size_t>(k*d)*sizeof(double));
    for(std::int64_t i=0;i<n;++i){
        const auto group=labels[i];if(group<0||group>=k)return 2;
        double* target=sums+group*d;const double* value=weighted+i*d;
        for(std::int64_t c=0;c<d;++c)target[c]+=value[c];
    }
    return 0;
}
