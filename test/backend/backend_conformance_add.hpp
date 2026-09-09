#pragma once
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include "iom/tensor.hpp"

namespace iom_conformance {
namespace add_oracle {
struct Spec { unsigned e, f; int bias; bool inf; bool finite; };
inline Spec spec(iom::DataType t) noexcept {
 switch(t){case iom::DataType::F4_E2M1:return{2,1,1,0,1};case iom::DataType::F6_E2M3:return{2,3,1,0,1};case iom::DataType::F6_E3M2:return{3,2,3,0,1};case iom::DataType::F8_E4M3FN:return{4,3,7,0,1};case iom::DataType::F8_E5M2:return{5,2,15,1,0};case iom::DataType::F16:return{5,10,15,1,0};case iom::DataType::BF16:return{8,7,127,1,0};case iom::DataType::F32:return{8,23,127,1,0};default:return{11,52,1023,1,0};}
}
inline long double value(std::uint64_t r, Spec s) noexcept {
 const auto em=(std::uint64_t{1}<<s.e)-1, fm=(std::uint64_t{1}<<s.f)-1, ex=(r>>s.f)&em, fr=r&fm;
 if(ex==em && (!s.finite || s.e==4)){if(s.inf&&fr==0)return (r>>(s.e+s.f))?-INFINITY:INFINITY;return std::numeric_limits<long double>::quiet_NaN();}
 long double v=ex?std::ldexp((long double)((std::uint64_t{1}<<s.f)+fr),(int)ex-s.bias-(int)s.f):std::ldexp((long double)fr,1-s.bias-(int)s.f);
 return (r>>(s.e+s.f))?-v:v;
}
inline std::uint64_t round_encode(long double x, Spec s) noexcept {
 const std::uint64_t em=(std::uint64_t{1}<<s.e)-1, fm=(std::uint64_t{1}<<s.f)-1, sg=std::signbit(x)?std::uint64_t{1}:std::uint64_t{0}; x=std::fabs(x);
 if(std::isnan(x)) return (sg<<(s.e+s.f))|(s.finite ? (em<<s.f)|fm : (em<<s.f)|(std::uint64_t{1}<<(s.f-1)));
 if(std::isinf(x)) return (sg<<(s.e+s.f))|(s.inf?em<<s.f:(em-1<<s.f)|fm);
 if(x==0)return sg<<(s.e+s.f);
 int e=0; std::frexp(x,&e); --e; const int minsub=1-s.bias-(int)s.f, maxe=(int)(s.finite && s.e<4 ? em : em-1)-s.bias;
 auto rn=[](long double q){auto n=std::floor(q),r=q-n;return (std::uint64_t)(n+(r>0.5L||(r==0.5L&&((std::uint64_t)n&1))));};
 if(e<minsub+(int)s.f){auto q=rn(std::ldexp(x,-minsub));return (sg<<(s.e+s.f))|(q==(std::uint64_t{1}<<s.f)?std::uint64_t{1}<<s.f:q);}
 if(e>maxe)return (sg<<(s.e+s.f))|((s.finite && s.e<4 ? em : em-1)<<s.f)|fm;
 auto q=rn(std::ldexp(x,s.f-e)-(std::uint64_t{1}<<s.f));if(q==(std::uint64_t{1}<<s.f)){++e;q=0;}if(e>maxe)return (sg<<(s.e+s.f))|((s.finite && s.e<4 ? em : em-1)<<s.f)|fm;
 return (sg<<(s.e+s.f))|((std::uint64_t)(e+s.bias)<<s.f)|q;
}
inline std::uint64_t add(iom::DataType t,std::uint64_t a,std::uint64_t b) noexcept {
 unsigned w=0;switch(t){case iom::DataType::I2:case iom::DataType::U2:w=2;break;case iom::DataType::I4:case iom::DataType::U4:w=4;break;case iom::DataType::I8:case iom::DataType::U8:w=8;break;case iom::DataType::I16:case iom::DataType::U16:w=16;break;case iom::DataType::I32:case iom::DataType::U32:w=32;break;case iom::DataType::I64:case iom::DataType::U64:w=64;break;default:{auto s=spec(t);auto x=value(a,s),y=value(b,s);if(std::isnan(x)||std::isnan(y)||(std::isinf(x)&&std::isinf(y)&&std::signbit(x)!=std::signbit(y)))return round_encode(std::numeric_limits<long double>::quiet_NaN(),s);return round_encode(x+y,s);}}
 return (a+b)&(w==64?~std::uint64_t{}:((std::uint64_t{1}<<w)-1));
}
}
}
