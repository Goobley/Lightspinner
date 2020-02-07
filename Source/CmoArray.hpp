#ifndef CMO_ARRAY_HPP
#define CMO_ARRAY_HPP

#include <cstdint>
#include <vector>
#include <array>
#include <cstdio>
#include <cassert>
#include <utility>

namespace Jasnah
{
// NOTE(cmo): The standards commitee are very emphatic now that indices and
// lengths of arrays should be signed. This makes sense when differencing etc.
// and it looks like std2 will actually follow this convention. We may as well
// get on board now.
typedef int64_t i64;
#ifdef CMO_ARRAY_BOUNDS_CHECK
    #define DO_BOUNDS_CHECK() assert((i0 >= 0) && (i0 < dim0))
    #define DO_BOUNDS_CHECK_M1()
#else
    #define DO_BOUNDS_CHECK()
    #define DO_BOUNDS_CHECK_M1()
#endif

template <typename T> struct Array1Own;
template <typename T> struct Array2Own;
template <typename T> struct Array3Own;
template <typename T> struct Array4Own;
template <typename T> struct Array5Own;
template <typename T> struct Array2NonOwn;
template <typename T> struct Array3NonOwn;
template <typename T> struct Array4NonOwn;
template <typename T> struct Array5NonOwn;

template <typename T>
struct Array1NonOwn
{
    T* data;
    i64 Ndim;
    i64 dim0;
    Array1NonOwn() : data(nullptr), Ndim(1), dim0(0)
    {}
    Array1NonOwn(T* data_, i64 dim) : data(data_), Ndim(1), dim0(dim)
    {}
    Array1NonOwn(std::vector<T>* vec) : data(vec->data()), Ndim(1), dim0(vec->size())
    {}
    Array1NonOwn(const Array1NonOwn& other) = default;
    Array1NonOwn(Array1NonOwn&& other) = default;
    Array1NonOwn(Array1Own<T>& other) : data(other.data()), Ndim(other.Ndim), dim0(other.dim0)
    {}

    template<typename U = T, typename = std::enable_if_t<std::is_const<U>::value>>
    Array1NonOwn(const Array1NonOwn<typename std::remove_const<T>::type>& other) : data(other.data()), Ndim(other.Ndim), dim0(other.dim0)
    {}

    Array1NonOwn&
    operator=(const Array1NonOwn& other) = default;

    Array1NonOwn&
    operator=(Array1NonOwn&& other) = default;

    inline explicit operator bool() const
    {
        return dim0 != 0;
    }

    void fill(T val)
    {
        for (i64 i = 0; i < dim0; ++i)
        {
            data[i] = val;
        }
    }

    inline const i64* shape() const
    {
        return &dim0;
    }
    inline i64 shape(int i) const
    {
        assert(i >= 0 && i < Ndim);
        return dim0;
    }

    inline T& operator[](i64 i)
    {
        return data[i];
    }
    inline T operator[](i64 i) const
    {
        return data[i];
    }

    inline T& operator()(i64 i0)
    {
        DO_BOUNDS_CHECK();
        return data[i0];
    }
    inline T operator()(i64 i0) const
    {
        DO_BOUNDS_CHECK();
        return data[i0];
    }

    Array1NonOwn<T> slice(i64 start, i64 end)
    {
        assert(start >= 0 && start < dim0 && end <= dim0);

        return Array1NonOwn(data+start, end-start);
    }

    Array1NonOwn<const T> slice(i64 start, i64 end) const
    {
        assert(start >= 0 && start < dim0 && end <= dim0);

        return Array1NonOwn(data+start, end-start);
    }

    Array2NonOwn<T> reshape(i64 d0, i64 d1)
    {
        if (d0 * d1 != dim0)
        {
            printf("Cannot reshape array [%d] to [%d x %d]\n", dim0, d0, d1);
            assert(d0 * d1 == dim0);
        }
        return Array2NonOwn(data, d0, d1);
    }

    Array2NonOwn<const T> reshape(i64 d0, i64 d1) const
    {
        if (d0 * d1 != dim0)
        {
            printf("Cannot reshape array [%d] to [%d x %d]\n", dim0, d0, d1);
            assert(d0 * d1 == dim0);
        }
        return Array2NonOwn(data, d0, d1);
    }

    Array3NonOwn<T> reshape(i64 d0, i64 d1, i64 d2)
    {
        if (d0 * d1 * d2 != dim0)
        {
            printf("Cannot reshape array [%d] to [%d x %d x %d]\n", dim0, d0, d1, d2);
            assert(d0 * d1 * d2 == dim0);
        }
        return Array3NonOwn(data, d0, d1, d2);
    }

    Array3NonOwn<const T> reshape(i64 d0, i64 d1, i64 d2) const
    {
        if (d0 * d1 * d2 != dim0)
        {
            printf("Cannot reshape array [%d] to [%d x %d x %d]\n", dim0, d0, d1, d2);
            assert(d0 * d1 * d2 == dim0);
        }
        return Array3NonOwn(data, d0, d1, d2);
    }

    Array4NonOwn<T> reshape(i64 d0, i64 d1, i64 d2, i64 d3)
    {
        if (d0 * d1 * d2 * d3 != dim0)
        {
            printf("Cannot reshape array [%d] to [%d x %d x %d x %d]\n", dim0, d0, d1, d2, d3);
            assert(d0 * d1 * d2 * d3 == dim0);
        }
        return Array4NonOwn(data, d0, d1, d2, d3);
    }

    Array4NonOwn<const T> reshape(i64 d0, i64 d1, i64 d2, i64 d3) const
    {
        if (d0 * d1 * d2 * d3 != dim0)
        {
            printf("Cannot reshape array [%d] to [%d x %d x %d x %d]\n", dim0, d0, d1, d2, d3);
            assert(d0 * d1 * d2 * d3 == dim0);
        }
        return Array4NonOwn(data, d0, d1, d2, d3);
    }

    Array5NonOwn<T> reshape(i64 d0, i64 d1, i64 d2, i64 d3, i64 d4)
    {
        if (d0 * d1 * d2 * d3 * d4 != dim0)
        {
            printf("Cannot reshape array [%d] to [%d x %d x %d x %d x %d]\n", dim0, d0, d1, d2, d3, d4);
            assert(d0 * d1 * d2 * d3 * d4 == dim0);
        }
        return Array5NonOwn(data, d0, d1, d2, d3, d4);
    }

    Array5NonOwn<const T> reshape(i64 d0, i64 d1, i64 d2, i64 d3, i64 d4) const
    {
        if (d0 * d1 * d2 * d3 * d4 != dim0)
        {
            printf("Cannot reshape array [%d] to [%d x %d x %d x %d x %d]\n", dim0, d0, d1, d2, d3, d4);
            assert(d0 * d1 * d2 * d3 * d4 == dim0);
        }
        return Array5NonOwn(data, d0, d1, d2, d3, d4);
    }
};

template <typename T>
struct Array1Own
{
    std::vector<T> dataStore;
    i64 Ndim;
    i64 dim0;
    Array1Own() : dataStore(), Ndim(1), dim0(0)
    {}
    Array1Own(i64 size) : dataStore(size), Ndim(1), dim0(size)
    {}
    Array1Own(T val, i64 size) : dataStore(size, val), Ndim(1), dim0(size)
    {}
    Array1Own(const Array1NonOwn<T>& other) : dataStore(other.data, other.data + other.dim0), Ndim(other.Ndim), dim0(other.dim0)
    {}
    Array1Own(const Array1Own& other) = default;
    Array1Own(Array1Own&& other) = default;

    Array1Own&
    operator=(const Array1NonOwn<T>& other)
    {
        Ndim = other.Ndim;
        dim0 = other.dim0;
        dataStore.assign(other.data, other.data + other.dim0);
        return *this;
    }
    Array1Own&
    operator=(const Array1Own& other) = default;

    Array1Own&
    operator=(Array1Own&& other) = default;

    inline explicit operator bool() const
    {
        return dim0 != 0;
    }

    T* data()
    {
        return dataStore.data();
    }

    void fill(T val)
    {
        for (i64 i = 0; i < dim0; ++i)
        {
            dataStore[i] = val;
        }
    }

    inline const i64* shape() const
    {
        return &dim0;
    }
    inline i64 shape(int i) const
    {
        assert(i >= 0 && i < Ndim);
        return dim0;
    }

    inline T& operator[](i64 i)
    {
        return dataStore[i];
    }
    inline T operator[](i64 i) const
    {
        return dataStore[i];
    }

    inline T& operator()(i64 i0)
    {
        DO_BOUNDS_CHECK();
        return dataStore[i0];
    }
    inline T operator()(i64 i0) const
    {
        DO_BOUNDS_CHECK();
        return dataStore[i0];
    }

    Array1NonOwn<T> slice(i64 start, i64 end)
    {
        assert(start >= 0 && start < dim0 && end <= dim0);

        return Array1NonOwn(data()+start, end-start);
    }

    Array1NonOwn<const T> slice(i64 start, i64 end) const
    {
        assert(start >= 0 && start < dim0 && end <= dim0);

        return Array1NonOwn(data()+start, end-start);
    }

    Array2NonOwn<T> reshape(i64 d0, i64 d1)
    {
        if (d0 * d1 != dim0)
        {
            printf("Cannot reshape array [%d] to [%d x %d]\n", dim0, d0, d1);
            assert(d0 * d1 == dim0);
        }
        return Array2NonOwn(data(), d0, d1);
    }

    Array2NonOwn<const T> reshape(i64 d0, i64 d1) const
    {
        if (d0 * d1 != dim0)
        {
            printf("Cannot reshape array [%d] to [%d x %d]\n", dim0, d0, d1);
            assert(d0 * d1 == dim0);
        }
        return Array2NonOwn(data(), d0, d1);
    }

    Array3NonOwn<T> reshape(i64 d0, i64 d1, i64 d2)
    {
        if (d0 * d1 * d2 != dim0)
        {
            printf("Cannot reshape array [%d] to [%d x %d x %d]\n", dim0, d0, d1, d2);
            assert(d0 * d1 * d2 == dim0);
        }
        return Array3NonOwn(data(), d0, d1, d2);
    }

    Array3NonOwn<const T> reshape(i64 d0, i64 d1, i64 d2) const
    {
        if (d0 * d1 * d2 != dim0)
        {
            printf("Cannot reshape array [%d] to [%d x %d x %d]\n", dim0, d0, d1, d2);
            assert(d0 * d1 * d2 == dim0);
        }
        return Array3NonOwn(data(), d0, d1, d2);
    }

    Array4NonOwn<T> reshape(i64 d0, i64 d1, i64 d2, i64 d3)
    {
        if (d0 * d1 * d2 * d3 != dim0)
        {
            printf("Cannot reshape array [%d] to [%d x %d x %d x %d]\n", dim0, d0, d1, d2, d3);
            assert(d0 * d1 * d2 * d3 == dim0);
        }
        return Array4NonOwn(data(), d0, d1, d2, d3);
    }

    Array4NonOwn<const T> reshape(i64 d0, i64 d1, i64 d2, i64 d3) const
    {
        if (d0 * d1 * d2 * d3 != dim0)
        {
            printf("Cannot reshape array [%d] to [%d x %d x %d x %d]\n", dim0, d0, d1, d2, d3);
            assert(d0 * d1 * d2 * d3 == dim0);
        }
        return Array4NonOwn(data(), d0, d1, d2, d3);
    }

    Array5NonOwn<T> reshape(i64 d0, i64 d1, i64 d2, i64 d3, i64 d4)
    {
        if (d0 * d1 * d2 * d3 * d4 != dim0)
        {
            printf("Cannot reshape array [%d] to [%d x %d x %d x %d x %d]\n", dim0, d0, d1, d2, d3, d4);
            assert(d0 * d1 * d2 * d3 * d4 == dim0);
        }
        return Array5NonOwn(data(), d0, d1, d2, d3, d4);
    }

    Array5NonOwn<const T> reshape(i64 d0, i64 d1, i64 d2, i64 d3, i64 d4) const
    {
        if (d0 * d1 * d2 * d3 * d4 != dim0)
        {
            printf("Cannot reshape array [%d] to [%d x %d x %d x %d x %d]\n", dim0, d0, d1, d2, d3, d4);
            assert(d0 * d1 * d2 * d3 * d4 == dim0);
        }
        return Array5NonOwn(data(), d0, d1, d2, d3, d4);
    }
};

#if defined(DO_BOUNDS_CHECK) && defined(CMO_ARRAY_BOUNDS_CHECK)
    #undef DO_BOUNDS_CHECK
    #undef DO_BOUNDS_CHECK_M1
    #define DO_BOUNDS_CHECK_M1() assert((i0 >= 0) && (i0 < dim[0]))
    #define DO_BOUNDS_CHECK() assert((i0 >= 0) && (i0 < dim[0]) && (i1 >= 0) && (i1 < dim[1]))
#endif

template <typename T>
struct Array2NonOwn
{
    T* data;
    i64 Ndim;
    std::array<i64, 2> dim;
    Array2NonOwn() : data(nullptr), Ndim(2), dim{}
    {}
    Array2NonOwn(T* data_, i64 dim0, i64 dim1) : data(data_), Ndim(2), dim{dim0, dim1}
    {}
    Array2NonOwn(const Array2NonOwn& other) = default;
    Array2NonOwn(Array2NonOwn&& other) = default;
    Array2NonOwn(Array2Own<T>& other) : data(other.data()), Ndim(other.Ndim), dim(other.dim)
    {}
    template<typename U = T, typename = std::enable_if_t<std::is_const<U>::value>>
    Array2NonOwn(const Array2NonOwn<typename std::remove_const<T>::type>& other) : data(other.data()), Ndim(other.Ndim), dim(other.dim)
    {}

    Array2NonOwn&
    operator=(const Array2NonOwn& other) = default;

    Array2NonOwn&
    operator=(Array2NonOwn&& other) = default;

    void fill(T val)
    {
        for (i64 i = 0; i < dim[0]*dim[1]; ++i)
        {
            data[i] = val;
        }
    }

    inline explicit operator bool() const
    {
        return dim[0] != 0;
    }

    inline const decltype(dim) shape() const
    {
        return dim;
    }
    inline i64 shape(int i) const
    {
        assert(i >= 0 && i < Ndim);
        return dim[i];
    }

    inline T& operator[](i64 i)
    {
        return data[i];
    }
    inline T operator[](i64 i) const
    {
        return data[i];
    }

    inline T& operator()(i64 i0, i64 i1)
    {
        DO_BOUNDS_CHECK();
        return data[i0*dim[1] + i1];
    }
    inline T operator()(i64 i0, i64 i1) const
    {
        DO_BOUNDS_CHECK();
        return data[i0*dim[1] + i1];
    }
    const T* const_slice(i64 i0) const
    {
        DO_BOUNDS_CHECK_M1();
        return &data[i0*dim[1]];
    }

    Array1NonOwn<T> operator()(i64 i0)
    {
        return Array1NonOwn<T>(&data[i0*dim[1]], dim[1]);
    }

    Array1NonOwn<const T> operator()(i64 i0) const
    {
        return Array1NonOwn<const T>(const_slice(i0), dim[1]);
    }

    Array1NonOwn<const T> flatten() const
    {
        return Array1NonOwn(data, dim[0]*dim[1]);
    }

    Array1NonOwn<T> flatten()
    {
        return Array1NonOwn(data, dim[0]*dim[1]);
    }

    Array1NonOwn<T> reshape(i64 d0)
    {
        if (d0 != dim[0] * dim[1])
        {
            printf("Cannot reshape array [%d x %d] to [%d]\n", dim[0], dim[1], d0);
            assert(d0 == dim[0] * dim[1]);
        }
        return Array1NonOwn(data, d0);
    }
    Array1NonOwn<const T> reshape(i64 d0) const
    {
        if (d0 != dim[0] * dim[1])
        {
            printf("Cannot reshape array [%d x %d] to [%d]\n", dim[0], dim[1], d0);
            assert(d0 == dim[0] * dim[1]);
        }
        return Array1NonOwn(data, d0);
    }

    Array3NonOwn<T> reshape(i64 d0, i64 d1, i64 d2)
    {
        if (d0 * d1 * d2 != dim[0] * dim[1])
        {
            printf("Cannot reshape array [%d x %d] to [%d x %d x %d]\n", dim[0], dim[1], d0, d1, d2);
            assert(d0 * d1 * d2 == dim[0] * dim[1]);
        }
        return Array3NonOwn(data, d0, d1, d2);
    }
    Array3NonOwn<const T> reshape(i64 d0, i64 d1, i64 d2) const
    {
        if (d0 * d1 * d2 != dim[0] * dim[1])
        {
            printf("Cannot reshape array [%d x %d] to [%d x %d x %d]\n", dim[0], dim[1], d0, d1, d2);
            assert(d0 * d1 * d2 == dim[0] * dim[1]);
        }
        return Array3NonOwn(data, d0, d1, d2);
    }

    Array4NonOwn<T> reshape(i64 d0, i64 d1, i64 d2, i64 d3)
    {
        if (d0 * d1 * d2 * d3 != dim[0] * dim[1])
        {
            printf("Cannot reshape array [%d x %d] to [%d x %d x %d x %d]\n", dim[0], dim[1], d0, d1, d2, d3);
            assert(d0 * d1 * d2 * d3 == dim[0] * dim[1]);
        }
        return Array4NonOwn(data, d0, d1, d2, d3);
    }
    Array4NonOwn<const T> reshape(i64 d0, i64 d1, i64 d2, i64 d3) const
    {
        if (d0 * d1 * d2 * d3 != dim[0] * dim[1])
        {
            printf("Cannot reshape array [%d x %d] to [%d x %d x %d x %d]\n", dim[0], dim[1], d0, d1, d2, d3);
            assert(d0 * d1 * d2 * d3 == dim[0] * dim[1]);
        }
        return Array4NonOwn(data, d0, d1, d2, d3);
    }

    Array5NonOwn<T> reshape(i64 d0, i64 d1, i64 d2, i64 d3, i64 d4)
    {
        if (d0 * d1 * d2 * d3 * d4 != dim[0] * dim[1])
        {
            printf("Cannot reshape array [%d x %d] to [%d x %d x %d x %d x %d]\n", dim[0], dim[1], d0, d1, d2, d3, d4);
            assert(d0 * d1 * d2 * d3 * d4 == dim[0] * dim[1]);
        }
        return Array5NonOwn(data, d0, d1, d2, d3, d4);
    }
    Array5NonOwn<const T> reshape(i64 d0, i64 d1, i64 d2, i64 d3, i64 d4) const
    {
        if (d0 * d1 * d2 * d3 * d4 != dim[0] * dim[1])
        {
            printf("Cannot reshape array [%d x %d] to [%d x %d x %d x %d x %d]\n", dim[0], dim[1], d0, d1, d2, d3, d4);
            assert(d0 * d1 * d2 * d3 * d4 == dim[0] * dim[1]);
        }
        return Array5NonOwn(data, d0, d1, d2, d3, d4);
    }
};

template <typename T>
struct Array2Own
{
    std::vector<T> dataStore;
    i64 Ndim;
    std::array<i64, 2> dim;
    Array2Own() : dataStore(), Ndim(2), dim{}
    {}
    Array2Own(i64 size1, i64 size2) : dataStore(size1*size2), Ndim(2), dim{size1, size2}
    {}
    Array2Own(T val, i64 size1, i64 size2) : dataStore(size1*size2, val), Ndim(2), dim{size1, size2}
    {}
    Array2Own(const Array2NonOwn<T>& other) : dataStore(other.data, other.data+other.dim[0]*other.dim[1]), Ndim(other.Ndim), dim(other.dim)
    {}
    Array2Own(const Array2Own& other) = default;
    Array2Own(Array2Own&& other) = default;
    
    Array2Own&
    operator=(const Array2NonOwn<T>& other)
    {
        Ndim = other.Ndim;
        dim = other.dim;
        auto len = dim[0]*dim[1];
        dataStore.assign(other.data, other.data+len);
        return *this;
    }

    Array2Own&
    operator=(const Array2Own& other) = default;

    Array2Own&
    operator=(Array2Own&& other) = default;

    inline explicit operator bool() const
    {
        return dim[0] != 0;
    }

    T* data()
    {
        return dataStore.data();
    }

    void fill(T val)
    {
        for (i64 i = 0; i < dim[0]*dim[1]; ++i)
        {
            dataStore[i] = val;
        }
    }

    inline const decltype(dim) shape() const
    {
        return dim;
    }
    inline i64 shape(int i) const
    {
        assert(i >= 0 && i < Ndim);
        return dim[i];
    }

    inline T& operator[](i64 i)
    {
        return dataStore[i];
    }
    inline T operator[](i64 i) const
    {
        return dataStore[i];
    }

    inline T& operator()(i64 i0, i64 i1)
    {
        DO_BOUNDS_CHECK();
        return dataStore[i0*dim[1] + i1];
    }
    inline T operator()(i64 i0, i64 i1) const
    {
        DO_BOUNDS_CHECK();
        return dataStore[i0*dim[1] + i1];
    }
    const T* const_slice(i64 i0) const
    {
        DO_BOUNDS_CHECK_M1();
        return &dataStore[i0*dim[1]];
    }

    Array1NonOwn<T> operator()(i64 i0)
    {
        return Array1NonOwn<T>(&dataStore[i0*dim[1]], dim[1]);
    }

    Array1NonOwn<const T> operator()(i64 i0) const
    {
        return Array1NonOwn<const T>(const_slice(i0), dim[1]);
    }

    Array1NonOwn<const T> flatten() const
    {
        return Array1NonOwn(data(), dim[0]*dim[1]);
    }

    Array1NonOwn<T> flatten()
    {
        return Array1NonOwn(data(), dim[0]*dim[1]);
    }

    Array1NonOwn<T> reshape(i64 d0)
    {
        if (d0 != dim[0] * dim[1])
        {
            printf("Cannot reshape array [%d x %d] to [%d]\n", dim[0], dim[1], d0);
            assert(d0 == dim[0] * dim[1]);
        }
        return Array1NonOwn(data(), d0);
    }
    Array1NonOwn<const T> reshape(i64 d0) const
    {
        if (d0 != dim[0] * dim[1])
        {
            printf("Cannot reshape array [%d x %d] to [%d]\n", dim[0], dim[1], d0);
            assert(d0 == dim[0] * dim[1]);
        }
        return Array1NonOwn(data(), d0);
    }

    Array3NonOwn<T> reshape(i64 d0, i64 d1, i64 d2)
    {
        if (d0 * d1 * d2 != dim[0] * dim[1])
        {
            printf("Cannot reshape array [%d x %d] to [%d x %d x %d]\n", dim[0], dim[1], d0, d1, d2);
            assert(d0 * d1 * d2 == dim[0] * dim[1]);
        }
        return Array3NonOwn(data(), d0, d1, d2);
    }
    Array3NonOwn<const T> reshape(i64 d0, i64 d1, i64 d2) const
    {
        if (d0 * d1 * d2 != dim[0] * dim[1])
        {
            printf("Cannot reshape array [%d x %d] to [%d x %d x %d]\n", dim[0], dim[1], d0, d1, d2);
            assert(d0 * d1 * d2 == dim[0] * dim[1]);
        }
        return Array3NonOwn(data(), d0, d1, d2);
    }

    Array4NonOwn<T> reshape(i64 d0, i64 d1, i64 d2, i64 d3)
    {
        if (d0 * d1 * d2 * d3 != dim[0] * dim[1])
        {
            printf("Cannot reshape array [%d x %d] to [%d x %d x %d x %d]\n", dim[0], dim[1], d0, d1, d2, d3);
            assert(d0 * d1 * d2 * d3 == dim[0] * dim[1]);
        }
        return Array4NonOwn(data(), d0, d1, d2, d3);
    }
    Array4NonOwn<const T> reshape(i64 d0, i64 d1, i64 d2, i64 d3) const
    {
        if (d0 * d1 * d2 * d3 != dim[0] * dim[1])
        {
            printf("Cannot reshape array [%d x %d] to [%d x %d x %d x %d]\n", dim[0], dim[1], d0, d1, d2, d3);
            assert(d0 * d1 * d2 * d3 == dim[0] * dim[1]);
        }
        return Array4NonOwn(data(), d0, d1, d2, d3);
    }

    Array5NonOwn<T> reshape(i64 d0, i64 d1, i64 d2, i64 d3, i64 d4)
    {
        if (d0 * d1 * d2 * d3 * d4 != dim[0] * dim[1])
        {
            printf("Cannot reshape array [%d x %d] to [%d x %d x %d x %d x %d]\n", dim[0], dim[1], d0, d1, d2, d3, d4);
            assert(d0 * d1 * d2 * d3 * d4 == dim[0] * dim[1]);
        }
        return Array5NonOwn(data(), d0, d1, d2, d3, d4);
    }
    Array5NonOwn<const T> reshape(i64 d0, i64 d1, i64 d2, i64 d3, i64 d4) const
    {
        if (d0 * d1 * d2 * d3 * d4 != dim[0] * dim[1])
        {
            printf("Cannot reshape array [%d x %d] to [%d x %d x %d x %d x %d]\n", dim[0], dim[1], d0, d1, d2, d3, d4);
            assert(d0 * d1 * d2 * d3 * d4 == dim[0] * dim[1]);
        }
        return Array5NonOwn(data(), d0, d1, d2, d3, d4);
    }
};

#if defined(DO_BOUNDS_CHECK) && defined(CMO_ARRAY_BOUNDS_CHECK)
    #undef DO_BOUNDS_CHECK
    #undef DO_BOUNDS_CHECK_M1
    #define DO_BOUNDS_CHECK_M1() assert((i0 >= 0) && (i0 < dim[0]) && (i1 >= 0) && (i1 < dim[1]))
    #define DO_BOUNDS_CHECK() assert((i0 >= 0) && (i0 < dim[0]) && (i1 >= 0) && (i1 < dim[1]) && (i2 >= 0) && (i2 < dim[2]))
#endif

template <typename T>
struct Array3NonOwn
{
    T* data;
    i64 Ndim;
    std::array<i64, 3> dim;
    std::array<i64, 2> dimProd;
    Array3NonOwn() : data(nullptr), Ndim(3), dim{0}, dimProd{0}
    {}
    Array3NonOwn(T* data_, i64 dim0, i64 dim1, i64 dim2) : data(data_), Ndim(3), dim{dim0, dim1, dim2}, dimProd{dim1*dim2, dim2}
    {}
    Array3NonOwn(const Array3NonOwn& other) = default;
    Array3NonOwn(Array3NonOwn&& other) = default;
    Array3NonOwn(Array3Own<T>& other) : data(other.data()), Ndim(other.Ndim), dim(other.dim), dimProd(other.dimProd)
    {}
    template<typename U = T, typename = std::enable_if_t<std::is_const<U>::value>>
    Array3NonOwn(const Array3NonOwn<typename std::remove_const<T>::type>& other) : data(other.data()), Ndim(other.Ndim), dim(other.dim), dimProd(other.dimProd)
    {}

    Array3NonOwn&
    operator=(const Array3NonOwn& other) = default;

    Array3NonOwn&
    operator=(Array3NonOwn&& other) = default;

    inline explicit operator bool() const
    {
        return dim[0] != 0;
    }

    void fill(T val)
    {
        for (i64 i = 0; i < dimProd[0] * dim[0]; ++i)
        {
            data[i] = val;
        }
    }

    inline const decltype(dim) shape() const
    {
        return dim;
    }
    inline i64 shape(int i) const
    {
        assert (i >= 0 && i < Ndim);
        return dim[i];
    }

    inline T& operator[](i64 i)
    {
        return data[i];
    }
    inline T operator[](i64 i) const
    {
        return data[i];
    }

    inline T& operator()(i64 i0, i64 i1, i64 i2)
    {
        DO_BOUNDS_CHECK();
        return data[i0*dimProd[0] + i1*dimProd[1] + i2];
    }
    inline T operator()(i64 i0, i64 i1, i64 i2) const
    {
        DO_BOUNDS_CHECK();
        return data[i0*dimProd[0] + i1*dimProd[1] + i2];
    }

    const T* const_slice(i64 i0, i64 i1) const
    {
        DO_BOUNDS_CHECK_M1();
        return &data[i0*dimProd[0] + i1*dimProd[1]];
    }

    Array2NonOwn<T> operator()(i64 i0)
    {
        return Array2NonOwn<T>(&(*this)(i0, 0, 0), dim[1], dim[2]);
    }

    Array2NonOwn<const T> operator()(i64 i0) const
    {
        return Array2NonOwn<const T>(const_slice(i0, 0), dim[1], dim[2]);
    }

    Array1NonOwn<T> operator()(i64 i0, i64 i1)
    {
        return Array1NonOwn<T>(&(*this)(i0, i1, 0), dim[2]);
    }

    Array1NonOwn<const T> operator()(i64 i0, i64 i1) const
    {
        return Array1NonOwn<const T>(const_slice(i0, i1), dim[2]);
    }

    Array1NonOwn<T> flatten()
    {
        return Array1NonOwn(data, dim[0]*dim[1]*dim[2]);
    }
    Array1NonOwn<const T> flatten() const
    {
        return Array1NonOwn(data, dim[0]*dim[1]*dim[2]);
    }

    Array1NonOwn<T> reshape(i64 d0)
    {
        if (d0 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d] to [%d]\n", dim[0], dim[1], dim[2], d0);
            assert(d0 == dimProd[0] * dim[0]);
        }
        return Array1NonOwn(data, d0);
    }
    Array1NonOwn<const T> reshape(i64 d0) const
    {
        if (d0 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d] to [%d]\n", dim[0], dim[1], dim[2], d0);
            assert(d0 == dimProd[0] * dim[0]);
        }
        return Array1NonOwn(data, d0);
    }

    Array2NonOwn<T> reshape(i64 d0, i64 d1)
    {
        if (d0 * d1 != dim[0] * dim[1] * dim[2])
        {
            printf("Cannot reshape array [%d x %d x %d] to [%d x %d]\n", dim[0], dim[1], dim[2], d0, d1);
            assert(d0 * d1 == dimProd[0] * dim[0]);
        }
        return Array2NonOwn(data, d0, d1);
    }
    Array2NonOwn<const T> reshape(i64 d0, i64 d1) const
    {
        if (d0 * d1 != dim[0] * dim[1] * dim[2])
        {
            printf("Cannot reshape array [%d x %d x %d] to [%d x %d]\n", dim[0], dim[1], dim[2], d0, d1);
            assert(d0 * d1 == dimProd[0] * dim[0]);
        }
        return Array2NonOwn(data, d0, d1);
    }

    Array4NonOwn<T> reshape(i64 d0, i64 d1, i64 d2, i64 d3)
    {
        if (d0 * d1 * d2 * d3 != dim[0] * dim[1] * dim[2])
        {
            printf("Cannot reshape array [%d x %d x %d] to [%d x %d x %d x %d]\n", dim[0], dim[1], dim[2], d0, d1, d2, d3);
            assert(d0 * d1 * d2 * d3 == dimProd[0] * dim[0]);
        }
        return Array4NonOwn(data, d0, d1, d2, d3);
    }
    Array4NonOwn<const T> reshape(i64 d0, i64 d1, i64 d2, i64 d3) const
    {
        if (d0 * d1 * d2 * d3 != dim[0] * dim[1] * dim[2])
        {
            printf("Cannot reshape array [%d x %d x %d] to [%d x %d x %d x %d]\n", dim[0], dim[1], dim[2], d0, d1, d2, d3);
            assert(d0 * d1 * d2 * d3 == dimProd[0] * dim[0]);
        }
        return Array4NonOwn(data, d0, d1, d2, d3);
    }

    Array5NonOwn<T> reshape(i64 d0, i64 d1, i64 d2, i64 d3, i64 d4)
    {
        if (d0 * d1 * d2 * d3 * d4 != dim[0] * dim[1] * dim[2])
        {
            printf("Cannot reshape array [%d x %d x %d] to [%d x %d x %d x %d x %d]\n", dim[0], dim[1], dim[2], d0, d1, d2, d3, d4);
            assert(d0 * d1 * d2 * d3 * d4 == dimProd[0] * dim[0]);
        }
        return Array5NonOwn(data, d0, d1, d2, d3, d4);
    }
    Array5NonOwn<const T> reshape(i64 d0, i64 d1, i64 d2, i64 d3, i64 d4) const
    {
        if (d0 * d1 * d2 * d3 * d4 != dim[0] * dim[1] * dim[2])
        {
            printf("Cannot reshape array [%d x %d x %d] to [%d x %d x %d x %d x %d]\n", dim[0], dim[1], dim[2], d0, d1, d2, d3, d4);
            assert(d0 * d1 * d2 * d3 * d4 == dimProd[0] * dim[0]);
        }
        return Array5NonOwn(data, d0, d1, d2, d3, d4);
    }
};

template <typename T>
struct Array3Own
{
    std::vector<T> dataStore;
    i64 Ndim;
    std::array<i64, 3> dim;
    std::array<i64, 2> dimProd;
    Array3Own() : dataStore(), Ndim(3), dim{}, dimProd{}
    {}
    Array3Own(i64 dim0, i64 dim1, i64 dim2) : dataStore(dim0*dim1*dim2), Ndim(3), dim{dim0, dim1, dim2}, dimProd{dim1*dim2, dim2}
    {}
    Array3Own(T val, i64 dim0, i64 dim1, i64 dim2) : dataStore(dim0*dim1*dim2, val), Ndim(3), dim{dim0, dim1, dim2}, dimProd{dim1*dim2, dim2}
    {}
    Array3Own(const Array3NonOwn<T>& other) : dataStore(other.data, other.data+other.dim[0]*other.dim[1]*other.dim[2]),
                                              Ndim(other.Ndim), dim(other.dim), dimProd(other.dimProd)
    {}
    Array3Own(const Array3Own& other) = default;
    Array3Own(Array3Own&& other) = default;

    Array3Own&
    operator=(const Array3NonOwn<T>& other)
    {
        Ndim = other.Ndim;
        dim = other.dim;
        dimProd = other.dimProd;
        auto len = other.dim[0]*other.dim[1]*other.dim[2];
        dataStore.assign(other.data, other.data+len);
        return *this;
    }

    Array3Own&
    operator=(const Array3Own& other) = default;

    Array3Own&
    operator=(Array3Own&& other) = default;

    inline explicit operator bool() const
    {
        return dim[0] != 0;
    }

    T* data()
    {
        return dataStore.data();
    }

    void fill(T val)
    {
        for (i64 i = 0; i < dimProd[0] * dim[0]; ++i)
        {
            dataStore[i] = val;
        }
    }

    inline const decltype(dim) shape() const
    {
        return dim;
    }
    inline i64 shape(int i) const
    {
        assert (i >= 0 && i < Ndim);
        return dim[i];
    }

    inline T& operator[](i64 i)
    {
        return dataStore[i];
    }
    inline T operator[](i64 i) const
    {
        return dataStore[i];
    }

    inline T& operator()(i64 i0, i64 i1, i64 i2)
    {
        DO_BOUNDS_CHECK();
        return dataStore[i0*dimProd[0] + i1*dimProd[1] + i2];
    }
    inline T operator()(i64 i0, i64 i1, i64 i2) const
    {
        DO_BOUNDS_CHECK();
        return dataStore[i0*dimProd[0] + i1*dimProd[1] + i2];
    }

    const T* const_slice(i64 i0, i64 i1) const
    {
        DO_BOUNDS_CHECK_M1();
        return &dataStore[i0*dimProd[0] + i1*dimProd[1]];
    }

    Array2NonOwn<T> operator()(i64 i0)
    {
        return Array2NonOwn<T>(&(*this)(i0, 0, 0), dim[1], dim[2]);
    }

    Array2NonOwn<const T> operator()(i64 i0) const
    {
        return Array2NonOwn<const T>(const_slice(i0, 0), dim[1], dim[2]);
    }

    Array1NonOwn<T> operator()(i64 i0, i64 i1)
    {
        return Array1NonOwn<T>(&(*this)(i0, i1, 0), dim[2]);
    }

    Array1NonOwn<const T> operator()(i64 i0, i64 i1) const
    {
        return Array1NonOwn<const T>(const_slice(i0, i1), dim[2]);
    }

    Array1NonOwn<T> flatten()
    {
        return Array1NonOwn(data(), dim[0]*dim[1]*dim[2]);
    }
    Array1NonOwn<const T> flatten() const
    {
        return Array1NonOwn(data(), dim[0]*dim[1]*dim[2]);
    }

    Array1NonOwn<T> reshape(i64 d0)
    {
        if (d0 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d] to [%d]\n", dim[0], dim[1], dim[2], d0);
            assert(d0 == dimProd[0] * dim[0]);
        }
        return Array1NonOwn(data(), d0);
    }
    Array1NonOwn<const T> reshape(i64 d0) const
    {
        if (d0 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d] to [%d]\n", dim[0], dim[1], dim[2], d0);
            assert(d0 == dimProd[0] * dim[0]);
        }
        return Array1NonOwn(data(), d0);
    }

    Array2NonOwn<T> reshape(i64 d0, i64 d1)
    {
        if (d0 * d1 != dim[0] * dim[1] * dim[2])
        {
            printf("Cannot reshape array [%d x %d x %d] to [%d x %d]\n", dim[0], dim[1], dim[2], d0, d1);
            assert(d0 * d1 == dimProd[0] * dim[0]);
        }
        return Array2NonOwn(data(), d0, d1);
    }
    Array2NonOwn<const T> reshape(i64 d0, i64 d1) const
    {
        if (d0 * d1 != dim[0] * dim[1] * dim[2])
        {
            printf("Cannot reshape array [%d x %d x %d] to [%d x %d]\n", dim[0], dim[1], dim[2], d0, d1);
            assert(d0 * d1 == dimProd[0] * dim[0]);
        }
        return Array2NonOwn(data(), d0, d1);
    }

    Array4NonOwn<T> reshape(i64 d0, i64 d1, i64 d2, i64 d3)
    {
        if (d0 * d1 * d2 * d3 != dim[0] * dim[1] * dim[2])
        {
            printf("Cannot reshape array [%d x %d x %d] to [%d x %d x %d x %d]\n", dim[0], dim[1], dim[2], d0, d1, d2, d3);
            assert(d0 * d1 * d2 * d3 == dimProd[0] * dim[0]);
        }
        return Array4NonOwn(data(), d0, d1, d2, d3);
    }
    Array4NonOwn<const T> reshape(i64 d0, i64 d1, i64 d2, i64 d3) const
    {
        if (d0 * d1 * d2 * d3 != dim[0] * dim[1] * dim[2])
        {
            printf("Cannot reshape array [%d x %d x %d] to [%d x %d x %d x %d]\n", dim[0], dim[1], dim[2], d0, d1, d2, d3);
            assert(d0 * d1 * d2 * d3 == dimProd[0] * dim[0]);
        }
        return Array4NonOwn(data(), d0, d1, d2, d3);
    }

    Array5NonOwn<T> reshape(i64 d0, i64 d1, i64 d2, i64 d3, i64 d4)
    {
        if (d0 * d1 * d2 * d3 * d4 != dim[0] * dim[1] * dim[2])
        {
            printf("Cannot reshape array [%d x %d x %d] to [%d x %d x %d x %d x %d]\n", dim[0], dim[1], dim[2], d0, d1, d2, d3, d4);
            assert(d0 * d1 * d2 * d3 * d4 == dimProd[0] * dim[0]);
        }
        return Array5NonOwn(data(), d0, d1, d2, d3, d4);
    }
    Array5NonOwn<const T> reshape(i64 d0, i64 d1, i64 d2, i64 d3, i64 d4) const
    {
        if (d0 * d1 * d2 * d3 * d4 != dim[0] * dim[1] * dim[2])
        {
            printf("Cannot reshape array [%d x %d x %d] to [%d x %d x %d x %d x %d]\n", dim[0], dim[1], dim[2], d0, d1, d2, d3, d4);
            assert(d0 * d1 * d2 * d3 * d4 == dimProd[0] * dim[0]);
        }
        return Array5NonOwn(data(), d0, d1, d2, d3, d4);
    }
};

#if defined(DO_BOUNDS_CHECK) && defined(CMO_ARRAY_BOUNDS_CHECK)
    #undef DO_BOUNDS_CHECK
    #undef DO_BOUNDS_CHECK_M1
    #define DO_BOUNDS_CHECK_M1() assert((i0 >= 0) && (i0 < dim[0]) && (i1 >= 0) && (i1 < dim[1]) && (i2 >= 0) && (i2 < dim[2]))
    #define DO_BOUNDS_CHECK() assert((i0 >= 0) && (i0 < dim[0]) && (i1 >= 0) && (i1 < dim[1]) && (i2 >= 0) && (i2 < dim[2]) && (i3 >= 0) && (i3 < dim[3]))
#endif

template <typename T>
struct Array4NonOwn
{
    T* data;
    i64 Ndim;
    std::array<i64, 4> dim;
    std::array<i64, 3> dimProd;
    Array4NonOwn() : data(nullptr), Ndim(4), dim{0}, dimProd{0}
    {}
    Array4NonOwn(T* data_, i64 dim0, i64 dim1, i64 dim2, i64 dim3) : data(data_), Ndim(4), dim{dim0, dim1, dim2, dim3}, 
                                                                                 dimProd{dim1*dim2*dim3, dim2*dim3, dim3}
    {}
    Array4NonOwn(const Array4NonOwn& other) = default;
    Array4NonOwn(Array4NonOwn&& other) = default;
    Array4NonOwn(Array4Own<T>& other) : data(other.data()), Ndim(other.Ndim), dim(other.dim), dimProd(other.dimProd)
    {}
    template<typename U = T, typename = std::enable_if_t<std::is_const<U>::value>>
    Array4NonOwn(const Array4NonOwn<typename std::remove_const<T>::type>& other) : data(other.data()), Ndim(other.Ndim), dim(other.dim), dimProd(other.dimProd)
    {}

    Array4NonOwn&
    operator=(const Array4NonOwn& other) = default;

    Array4NonOwn&
    operator=(Array4NonOwn&& other) = default;

    inline explicit operator bool() const
    {
        return dim[0] != 0;
    }

    void fill(T val)
    {
        for (i64 i = 0; i < dimProd[0] * dim[0]; ++i)
        {
            data[i] = val;
        }
    }

    inline const decltype(dim) shape() const
    {
        return dim;
    }
    inline i64 shape(int i) const
    {
        assert(i >= 0 && i < Ndim);
        return dim[i];
    }

    inline T& operator[](i64 i)
    {
        return data[i];
    }
    inline T operator[](i64 i) const
    {
        return data[i];
    }

    inline T& operator()(i64 i0, i64 i1, i64 i2, i64 i3)
    {
        DO_BOUNDS_CHECK();
        return data[i0*dimProd[0] + i1*dimProd[1] + i2*dimProd[2] + i3];
    }

    inline T operator()(i64 i0, i64 i1, i64 i2, i64 i3) const
    {
        DO_BOUNDS_CHECK();
        return data[i0*dimProd[0] + i1*dimProd[1] + i2*dimProd[2] + i3];
    }

    const T* const_slice(i64 i0, i64 i1, i64 i2) const
    {
        DO_BOUNDS_CHECK_M1();
        return &data[i0*dimProd[0] + i1*dimProd[1] + i2*dimProd[2]];
    }

    Array3NonOwn<T> operator()(i64 i0)
    {
        return Array3NonOwn<T>(&(*this)(i0, 0, 0, 0), dim[1], dim[2], dim[3]);
    }

    Array3NonOwn<const T> operator()(i64 i0) const
    {
        return Array3NonOwn<T>(const_slice(i0, 0, 0), dim[1], dim[2], dim[3]);
    }

    Array2NonOwn<T> operator()(i64 i0, i64 i1)
    {
        return Array2NonOwn<T>(&(*this)(i0, i1, 0, 0), dim[2], dim[3]);
    }

    Array2NonOwn<const T> operator()(i64 i0, i64 i1) const
    {
        return Array2NonOwn<T>(const_slice(i0, i1, 0), dim[2], dim[3]);
    }

    Array1NonOwn<T> operator()(i64 i0, i64 i1, i64 i2)
    {
        return Array1NonOwn<T>(&(*this)(i0, i1, i2, 0), dim[3]);
    }

    Array1NonOwn<const T> operator()(i64 i0, i64 i1, i64 i2) const
    {
        return Array1NonOwn<const T>(const_slice(i0, i1, i2), dim[3]);
    }

    Array1NonOwn<T> flatten()
    {
        return Array1NonOwn(data, dimProd[0] * dim[0]);
    }
    Array1NonOwn<const T> flatten() const
    {
        return Array1NonOwn(data, dimProd[0] * dim[0]);
    }

    Array1NonOwn<T> reshape(i64 d0)
    {
        if (d0 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d] to [%d]\n", dim[0], dim[1], dim[2], dim[3], d0);
            assert(d0 == dimProd[0] * dim[0]);
        }
        return Array1NonOwn(data, d0);
    }
    Array1NonOwn<const T> reshape(i64 d0) const
    {
        if (d0 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d] to [%d]\n", dim[0], dim[1], dim[2], dim[3], d0);
            assert(d0 == dimProd[0] * dim[0]);
        }
        return Array1NonOwn(data, d0);
    }

    Array2NonOwn<T> reshape(i64 d0, i64 d1)
    {
        if (d0 * d1 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d] to [%d x %d]\n", dim[0], dim[1], dim[2], dim[3], d0, d1);
            assert(d0 * d1 == dimProd[0] * dim[0]);
        }
        return Array2NonOwn(data, d0, d1);
    }
    Array2NonOwn<const T> reshape(i64 d0, i64 d1) const
    {
        if (d0 * d1 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d] to [%d x %d]\n", dim[0], dim[1], dim[2], dim[3], d0, d1);
            assert(d0 * d1 == dimProd[0] * dim[0]);
        }
        return Array2NonOwn(data, d0, d1);
    }

    Array3NonOwn<T> reshape(i64 d0, i64 d1, i64 d2)
    {
        if (d0 * d1 * d2 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d] to [%d x %d x %d]\n", dim[0], dim[1], dim[2], dim[3], d0, d1, d2);
            assert(d0 * d1 * d2 == dimProd[0] * dim[0]);
        }
        return Array3NonOwn(data, d0, d1, d2);
    }
    Array3NonOwn<const T> reshape(i64 d0, i64 d1, i64 d2) const
    {
        if (d0 * d1 * d2 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d] to [%d x %d x %d]\n", dim[0], dim[1], dim[2], dim[3], d0, d1, d2);
            assert(d0 * d1 * d2 == dimProd[0] * dim[0]);
        }
        return Array3NonOwn(data, d0, d1, d2);
    }

    Array5NonOwn<T> reshape(i64 d0, i64 d1, i64 d2, i64 d3, i64 d4)
    {
        if (d0 * d1 * d2 * d3 * d4 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d] to [%d x %d x %d x %d x %d]\n", dim[0], dim[1], dim[2], dim[3], d0, d1, d2, d3, d4);
            assert(d0 * d1 * d2 * d3 * d4 == dimProd[0] * dim[0]);
        }
        return Array5NonOwn(data, d0, d1, d2, d3, d4);
    }
    Array5NonOwn<const T> reshape(i64 d0, i64 d1, i64 d2, i64 d3, i64 d4) const
    {
        if (d0 * d1 * d2 * d3 * d4 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d] to [%d x %d x %d x %d x %d]\n", dim[0], dim[1], dim[2], dim[3], d0, d1, d2, d3, d4);
            assert(d0 * d1 * d2 * d3 * d4 == dimProd[0] * dim[0]);
        }
        return Array5NonOwn(data, d0, d1, d2, d3, d4);
    }
};

template <typename T>
struct Array4Own
{
    std::vector<T> dataStore;
    i64 Ndim;
    std::array<i64, 4> dim;
    std::array<i64, 3> dimProd;
    Array4Own() : dataStore(), Ndim(4), dim{}, dimProd{}
    {}
    Array4Own(i64 dim0, i64 dim1, i64 dim2, i64 dim3) : dataStore(dim0*dim1*dim2*dim3), Ndim(4), dim{dim0, dim1, dim2, dim3},
                                                                    dimProd{dim1*dim2*dim3, dim2*dim3, dim3}
    {}
    Array4Own(T val, i64 dim0, i64 dim1, i64 dim2, i64 dim3) : dataStore(dim0*dim1*dim2*dim3, val), Ndim(4), dim{dim0, dim1, dim2, dim3}, 
                                                                              dimProd{dim1*dim2*dim3, dim2*dim3, dim3}
    {}
    Array4Own(const Array4NonOwn<T>& other) : dataStore(other.data, other.data+other.dim[0]*other.dimProd[0]), 
                                              Ndim(other.Ndim), dim(other.dim), dimProd(other.dimProd)
    {}

    Array4Own(const Array4Own& other) = default;
    Array4Own(Array4Own&& other) = default;

    Array4Own&
    operator=(const Array4NonOwn<T>& other)
    {
        Ndim = other.Ndim;
        dim = other.dim;
        dimProd = other.dimProd;
        auto len = other.dimProd[0] * other.dim[0];
        dataStore.assign(other.data, other.data+len);
        return *this;
    }
    Array4Own&
    operator=(const Array4Own& other) = default;

    Array4Own&
    operator=(Array4Own&& other) = default;

    T* data()
    {
        return dataStore.data();
    }

    inline explicit operator bool() const
    {
        return dim[0] != 0;
    }

    void fill(T val)
    {
        for (i64 i = 0; i < dimProd[0] * dim[0]; ++i)
        {
            dataStore[i] = val;
        }
    }

    inline const decltype(dim) shape() const
    {
        return dim;
    }
    inline i64 shape(int i) const
    {
        assert(i >= 0 && i < Ndim);
        return dim[i];
    }

    inline T& operator[](i64 i)
    {
        return dataStore[i];
    }
    inline T operator[](i64 i) const
    {
        return dataStore[i];
    }

    inline T& operator()(i64 i0, i64 i1, i64 i2, i64 i3)
    {
        DO_BOUNDS_CHECK();
        return dataStore[i0*dimProd[0] + i1*dimProd[1] + i2*dimProd[2] + i3];
    }

    inline T operator()(i64 i0, i64 i1, i64 i2, i64 i3) const
    {
        DO_BOUNDS_CHECK();
        return dataStore[i0*dimProd[0] + i1*dimProd[1] + i2*dimProd[2] + i3];
    }

    const T* const_slice(i64 i0, i64 i1, i64 i2) const
    {
        DO_BOUNDS_CHECK_M1();
        return &dataStore[i0*dimProd[0] + i1*dimProd[1] + i2*dimProd[2]];
    }

    Array3NonOwn<T> operator()(i64 i0)
    {
        return Array3NonOwn<T>(&(*this)(i0, 0, 0, 0), dim[1], dim[2], dim[3]);
    }

    Array3NonOwn<const T> operator()(i64 i0) const
    {
        return Array3NonOwn<T>(const_slice(i0, 0, 0), dim[1], dim[2], dim[3]);
    }

    Array2NonOwn<T> operator()(i64 i0, i64 i1)
    {
        return Array2NonOwn<T>(&(*this)(i0, i1, 0, 0), dim[2], dim[3]);
    }

    Array2NonOwn<const T> operator()(i64 i0, i64 i1) const
    {
        return Array2NonOwn<T>(const_slice(i0, i1, 0), dim[2], dim[3]);
    }

    Array1NonOwn<T> operator()(i64 i0, i64 i1, i64 i2)
    {
        return Array1NonOwn<T>(&(*this)(i0, i1, i2, 0), dim[3]);
    }

    Array1NonOwn<const T> operator()(i64 i0, i64 i1, i64 i2) const
    {
        return Array1NonOwn<const T>(const_slice(i0, i1, i2), dim[3]);
    }

    Array1NonOwn<T> flatten()
    {
        return Array1NonOwn(data(), dimProd[0] * dim[0]);
    }
    Array1NonOwn<const T> flatten() const
    {
        return Array1NonOwn(data(), dimProd[0] * dim[0]);
    }

    Array1NonOwn<T> reshape(i64 d0)
    {
        if (d0 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d] to [%d]\n", dim[0], dim[1], dim[2], dim[3], d0);
            assert(d0 == dimProd[0] * dim[0]);
        }
        return Array1NonOwn(data(), d0);
    }
    Array1NonOwn<const T> reshape(i64 d0) const
    {
        if (d0 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d] to [%d]\n", dim[0], dim[1], dim[2], dim[3], d0);
            assert(d0 == dimProd[0] * dim[0]);
        }
        return Array1NonOwn(data(), d0);
    }

    Array2NonOwn<T> reshape(i64 d0, i64 d1)
    {
        if (d0 * d1 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d] to [%d x %d]\n", dim[0], dim[1], dim[2], dim[3], d0, d1);
            assert(d0 * d1 == dimProd[0] * dim[0]);
        }
        return Array2NonOwn(data(), d0, d1);
    }
    Array2NonOwn<const T> reshape(i64 d0, i64 d1) const
    {
        if (d0 * d1 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d] to [%d x %d]\n", dim[0], dim[1], dim[2], dim[3], d0, d1);
            assert(d0 * d1 == dimProd[0] * dim[0]);
        }
        return Array2NonOwn(data(), d0, d1);
    }

    Array3NonOwn<T> reshape(i64 d0, i64 d1, i64 d2)
    {
        if (d0 * d1 * d2 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d] to [%d x %d x %d]\n", dim[0], dim[1], dim[2], dim[3], d0, d1, d2);
            assert(d0 * d1 * d2 == dimProd[0] * dim[0]);
        }
        return Array3NonOwn(data(), d0, d1, d2);
    }
    Array3NonOwn<const T> reshape(i64 d0, i64 d1, i64 d2) const
    {
        if (d0 * d1 * d2 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d] to [%d x %d x %d]\n", dim[0], dim[1], dim[2], dim[3], d0, d1, d2);
            assert(d0 * d1 * d2 == dimProd[0] * dim[0]);
        }
        return Array3NonOwn(data(), d0, d1, d2);
    }

    Array5NonOwn<T> reshape(i64 d0, i64 d1, i64 d2, i64 d3, i64 d4)
    {
        if (d0 * d1 * d2 * d3 * d4 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d] to [%d x %d x %d x %d x %d]\n", dim[0], dim[1], dim[2], dim[3], d0, d1, d2, d3, d4);
            assert(d0 * d1 * d2 * d3 * d4 == dimProd[0] * dim[0]);
        }
        return Array5NonOwn(data(), d0, d1, d2, d3, d4);
    }
    Array5NonOwn<const T> reshape(i64 d0, i64 d1, i64 d2, i64 d3, i64 d4) const
    {
        if (d0 * d1 * d2 * d3 * d4 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d] to [%d x %d x %d x %d x %d]\n", dim[0], dim[1], dim[2], dim[3], d0, d1, d2, d3, d4);
            assert(d0 * d1 * d2 * d3 * d4 == dimProd[0] * dim[0]);
        }
        return Array5NonOwn(data(), d0, d1, d2, d3, d4);
    }
};

#if defined(DO_BOUNDS_CHECK) && defined(CMO_ARRAY_BOUNDS_CHECK)
    #undef DO_BOUNDS_CHECK
    #undef DO_BOUNDS_CHECK_M1
    #define DO_BOUNDS_CHECK_M1() assert((i0 >= 0) && (i0 < dim[0]) && (i1 >= 0) && (i1 < dim[1]) && (i2 >= 0) && (i2 < dim[2]) && (i3 >= 0) && (i3 < dim[3]))
    #define DO_BOUNDS_CHECK() assert((i0 >= 0) && (i0 < dim[0]) && (i1 >= 0) && (i1 < dim[1]) && (i2 >= 0) && (i2 < dim[2]) && (i3 >= 0) && (i3 < dim[3]) && (i4 >= 0) && (i4 < dim[4]))
#endif

template <typename T>
struct Array5NonOwn
{
    T* data;
    i64 Ndim;
    std::array<i64, 5> dim;
    std::array<i64, 4> dimProd;
    Array5NonOwn() : data(nullptr), Ndim(5), dim{0}, dimProd{0}
    {}
    Array5NonOwn(T* data_, i64 dim0, i64 dim1, i64 dim2, i64 dim3, i64 dim4) : data(data_), Ndim(5), dim{dim0,dim1,dim2,dim3,dim4}, 
                                                                                              dimProd{dim1*dim2*dim3*dim4, dim2*dim3*dim4, dim3*dim4, dim4}
    {}
    Array5NonOwn(const Array5NonOwn& other) = default;
    Array5NonOwn(Array5NonOwn&& other) = default;
    Array5NonOwn(Array5Own<T>& other) : data(other.data()), Ndim(other.Ndim), dim(other.dim), dimProd(other.dimProd)
    {}
    template<typename U = T, typename = std::enable_if_t<std::is_const<U>::value>>
    Array5NonOwn(const Array5NonOwn<typename std::remove_const<T>::type>& other) : data(other.data()), Ndim(other.Ndim), dim(other.dim), dimProd(other.dimProd)
    {}

    Array5NonOwn&
    operator=(const Array5NonOwn& other) = default;

    Array5NonOwn&
    operator=(Array5NonOwn&& other) = default;

    inline explicit operator bool() const
    {
        return dim[0] != 0;
    }

    void fill(T val)
    {
        for (i64 i = 0; i < dimProd[0] * dim[0]; ++i)
        {
            data[i] = val;
        }
    }

    inline const decltype(dim) shape() const
    {
        return dim;
    }
    inline i64 shape(int i) const
    {
        assert(i >= 0 && i < Ndim);
        return dim[i];
    }

    inline T& operator[](i64 i)
    {
        return data[i];
    }
    inline T operator[](i64 i) const
    {
        return data[i];
    }

    inline T& operator()(i64 i0, i64 i1, i64 i2, i64 i3, i64 i4)
    {
        DO_BOUNDS_CHECK();
        return data[i0*dimProd[0] + i1*dimProd[1] + i2*dimProd[2] + i3*dimProd[3] + i4];
    }
    inline T operator()(i64 i0, i64 i1, i64 i2, i64 i3, i64 i4) const
    {
        DO_BOUNDS_CHECK();
        return data[i0*dimProd[0] + i1*dimProd[1] + i2*dimProd[2] + i3*dimProd[3] + i4];
    }

    const T* const_slice(i64 i0, i64 i1, i64 i2, i64 i3) const
    {
        DO_BOUNDS_CHECK_M1();
        return &data[i0*dimProd[0] + i1*dimProd[1] + i2*dimProd[2] + i3*dimProd[3]];
    }

    Array4NonOwn<T> operator()(i64 i0)
    {
        return Array4NonOwn<T>(&(*this)(i0, 0, 0, 0, 0), dim[1], dim[2], dim[3], dim[4]);
    }
    Array4NonOwn<const T> operator()(i64 i0) const
    {
        return Array4NonOwn<const T>(const_slice(i0, 0, 0, 0), dim[1], dim[2], dim[3], dim[4]);
    }

    Array3NonOwn<T> operator()(i64 i0, i64 i1)
    {
        return Array3NonOwn<T>(&(*this)(i0, i1, 0, 0, 0), dim[2], dim[3], dim[4]);
    }

    Array3NonOwn<const T> operator()(i64 i0, i64 i1) const
    {
        return Array3NonOwn<const T>(const_slice(i0, i1, 0, 0), dim[2], dim[3], dim[4]);
    }

    Array2NonOwn<T> operator()(i64 i0, i64 i1, i64 i2)
    {
        return Array2NonOwn<T>(&(*this)(i0, i1, i2, 0, 0), dim[3], dim[4]);
    }

    Array2NonOwn<const T> operator()(i64 i0, i64 i1, i64 i2) const
    {
        return Array2NonOwn<const T>(const_slice(i0, i1, i2, 0), dim[3], dim[4]);
    }

    Array1NonOwn<T> operator()(i64 i0, i64 i1, i64 i2, i64 i3)
    {
        return Array1NonOwn<T>(&(*this)(i0, i1, i2, i3, 0), dim[4]);
    }

    Array1NonOwn<const T> operator()(i64 i0, i64 i1, i64 i2, i64 i3) const
    {
        return Array1NonOwn<const T>(const_slice(i0, i1, i2, i3), dim[4]);
    }

    Array1NonOwn<T> flatten()
    {
        return Array1NonOwn(data, dimProd[0] * dim[0]);
    }
    Array1NonOwn<const T> flatten() const
    {
        return Array1NonOwn(data, dimProd[0] * dim[0]);
    }

    Array1NonOwn<T> reshape(i64 d0)
    {
        if (d0 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d x %d] to [%d]\n", dim[0], dim[1], dim[2], dim[3], dim[4], d0);
            assert(d0 == dimProd[0] * dim[0]);
        }
        return Array1NonOwn(data, d0);
    }
    Array1NonOwn<const T> reshape(i64 d0) const
    {
        if (d0 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d x %d] to [%d]\n", dim[0], dim[1], dim[2], dim[3], dim[4], d0);
            assert(d0 == dimProd[0] * dim[0]);
        }
        return Array1NonOwn(data, d0);
    }

    Array2NonOwn<T> reshape(i64 d0, i64 d1)
    {
        if (d0 * d1 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d x %d] to [%d x %d]\n", dim[0], dim[1], dim[2], dim[3], dim[4], d0, d1);
            assert(d0 * d1 == dimProd[0] * dim[0]);
        }
        return Array2NonOwn(data, d0, d1);
    }
    Array2NonOwn<const T> reshape(i64 d0, i64 d1) const
    {
        if (d0 * d1 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d x %d] to [%d x %d]\n", dim[0], dim[1], dim[2], dim[3], dim[4], d0, d1);
            assert(d0 * d1 == dimProd[0] * dim[0]);
        }
        return Array2NonOwn(data, d0, d1);
    }

    Array3NonOwn<T> reshape(i64 d0, i64 d1, i64 d2)
    {
        if (d0 * d1 * d2 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d x %d] to [%d x %d x %d]\n", dim[0], dim[1], dim[2], dim[3], dim[4], d0, d1, d2);
            assert(d0 * d1 * d2 == dimProd[0] * dim[0]);
        }
        return Array3NonOwn(data, d0, d1, d2);
    }
    Array3NonOwn<const T> reshape(i64 d0, i64 d1, i64 d2) const
    {
        if (d0 * d1 * d2 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d x %d] to [%d x %d x %d]\n", dim[0], dim[1], dim[2], dim[3], dim[4], d0, d1, d2);
            assert(d0 * d1 * d2 == dimProd[0] * dim[0]);
        }
        return Array3NonOwn(data, d0, d1, d2);
    }

    Array4NonOwn<T> reshape(i64 d0, i64 d1, i64 d2, i64 d3)
    {
        if (d0 * d1 * d2 * d3 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d x %d] to [%d x %d x %d x %d]\n", dim[0], dim[1], dim[2], dim[3], dim[4], d0, d1, d2, d3);
            assert(d0 * d1 * d2 * d3 == dimProd[0] * dim[0]);
        }
        return Array4NonOwn(data, d0, d1, d2, d3);
    }
    Array4NonOwn<const T> reshape(i64 d0, i64 d1, i64 d2, i64 d3) const
    {
        if (d0 * d1 * d2 * d3 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d x %d] to [%d x %d x %d x %d]\n", dim[0], dim[1], dim[2], dim[3], dim[4], d0, d1, d2, d3);
            assert(d0 * d1 * d2 * d3 == dimProd[0] * dim[0]);
        }
        return Array4NonOwn(data, d0, d1, d2, d3);
    }
};

template <typename T>
struct Array5Own
{
    std::vector<T> dataStore;
    i64 Ndim;
    std::array<i64, 5> dim;
    std::array<i64, 4> dimProd;
    Array5Own() : dataStore(), Ndim(5), dim{}, dimProd{}
    {}
    Array5Own(i64 d0, i64 d1, i64 d2, i64 d3, i64 d4) : dataStore(d0*d1*d2*d3*d4), Ndim(5), dim{d0,d1,d2,d3,d4}, 
                                                                       dimProd{d1*d2*d3*d4, d2*d3*d4, d3*d4, d4}
    {}
    Array5Own(T val, i64 d0, i64 d1, i64 d2, i64 d3, i64 d4) : dataStore(d0*d1*d2*d3*d4, val), Ndim(5), dim{d0,d1,d2,d3,d4}, 
                                                                              dimProd{d1*d2*d3*d4, d2*d3*d4, d3*d4, d4}
    {}
    Array5Own(const Array5NonOwn<T>& other) : dataStore(other.data, other.data+other.dim[0]*other.dimProd[0]), Ndim(other.Ndim), 
                                              dim(other.dim), dimProd(other.dimProd)
    {}
    Array5Own(const Array5Own& other) = default;
    Array5Own(Array5Own&& other) = default;

    Array5Own&
    operator=(const Array5NonOwn<T>& other)
    {
        Ndim = other.Ndim;
        dim = other.dim;
        dimProd = other.dimProd;
        dataStore.assign(other.data, other.data+other.dim[0]*other.dimProd[0]);
        return *this;
    }

    Array5Own&
    operator=(const Array5Own& other) = default;

    Array5Own&
    operator=(Array5Own&& other) = default;

    inline explicit operator bool() const
    {
        return dim[0] != 0;
    }

    T* data()
    {
        return dataStore.data();
    }

    void fill(T val)
    {
        for (i64 i = 0; i < dimProd[0] * dim[0]; ++i)
        {
            dataStore[i] = val;
        }
    }

    inline const decltype(dim) shape() const
    {
        return dim;
    }
    inline i64 shape(int i) const
    {
        assert(i >= 0 && i < Ndim);
        return dim[i];
    }

    inline T& operator[](i64 i)
    {
        return dataStore[i];
    }
    inline T operator[](i64 i) const
    {
        return dataStore[i];
    }

    inline T& operator()(i64 i0, i64 i1, i64 i2, i64 i3, i64 i4)
    {
        DO_BOUNDS_CHECK();
        return dataStore[i0*dimProd[0] + i1*dimProd[1] + i2*dimProd[2] + i3*dimProd[3] + i4];
    }
    inline T operator()(i64 i0, i64 i1, i64 i2, i64 i3, i64 i4) const
    {
        DO_BOUNDS_CHECK();
        return dataStore[i0*dimProd[0] + i1*dimProd[1] + i2*dimProd[2] + i3*dimProd[3] + i4];
    }

    const T* const_slice(i64 i0, i64 i1, i64 i2, i64 i3) const
    {
        DO_BOUNDS_CHECK_M1();
        return &dataStore[i0*dimProd[0] + i1*dimProd[1] + i2*dimProd[2] + i3*dimProd[3]];
    }

    Array4NonOwn<T> operator()(i64 i0)
    {
        return Array4NonOwn<T>(&(*this)(i0, 0, 0, 0, 0), dim[1], dim[2], dim[3], dim[4]);
    }
    Array4NonOwn<const T> operator()(i64 i0) const
    {
        return Array4NonOwn<const T>(const_slice(i0, 0, 0, 0), dim[1], dim[2], dim[3], dim[4]);
    }

    Array3NonOwn<T> operator()(i64 i0, i64 i1)
    {
        return Array3NonOwn<T>(&(*this)(i0, i1, 0, 0, 0), dim[2], dim[3], dim[4]);
    }

    Array3NonOwn<const T> operator()(i64 i0, i64 i1) const
    {
        return Array3NonOwn<const T>(const_slice(i0, i1, 0, 0), dim[2], dim[3], dim[4]);
    }

    Array2NonOwn<T> operator()(i64 i0, i64 i1, i64 i2)
    {
        return Array2NonOwn<T>(&(*this)(i0, i1, i2, 0, 0), dim[3], dim[4]);
    }

    Array2NonOwn<const T> operator()(i64 i0, i64 i1, i64 i2) const
    {
        return Array2NonOwn<const T>(const_slice(i0, i1, i2, 0), dim[3], dim[4]);
    }

    Array1NonOwn<T> operator()(i64 i0, i64 i1, i64 i2, i64 i3)
    {
        return Array1NonOwn<T>(&(*this)(i0, i1, i2, i3, 0), dim[4]);
    }

    Array1NonOwn<const T> operator()(i64 i0, i64 i1, i64 i2, i64 i3) const
    {
        return Array1NonOwn<const T>(const_slice(i0, i1, i2, i3), dim[4]);
    }

    Array1NonOwn<T> flatten()
    {
        return Array1NonOwn(data(), dimProd[0] * dim[0]);
    }
    Array1NonOwn<const T> flatten() const
    {
        return Array1NonOwn(data(), dimProd[0] * dim[0]);
    }

    Array1NonOwn<T> reshape(i64 d0)
    {
        if (d0 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d x %d] to [%d]\n", dim[0], dim[1], dim[2], dim[3], dim[4], d0);
            assert(d0 == dimProd[0] * dim[0]);
        }
        return Array1NonOwn(data(), d0);
    }
    Array1NonOwn<const T> reshape(i64 d0) const
    {
        if (d0 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d x %d] to [%d]\n", dim[0], dim[1], dim[2], dim[3], dim[4], d0);
            assert(d0 == dimProd[0] * dim[0]);
        }
        return Array1NonOwn(data(), d0);
    }

    Array2NonOwn<T> reshape(i64 d0, i64 d1)
    {
        if (d0 * d1 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d x %d] to [%d x %d]\n", dim[0], dim[1], dim[2], dim[3], dim[4], d0, d1);
            assert(d0 * d1 == dimProd[0] * dim[0]);
        }
        return Array2NonOwn(data(), d0, d1);
    }
    Array2NonOwn<const T> reshape(i64 d0, i64 d1) const
    {
        if (d0 * d1 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d x %d] to [%d x %d]\n", dim[0], dim[1], dim[2], dim[3], dim[4], d0, d1);
            assert(d0 * d1 == dimProd[0] * dim[0]);
        }
        return Array2NonOwn(data(), d0, d1);
    }

    Array3NonOwn<T> reshape(i64 d0, i64 d1, i64 d2)
    {
        if (d0 * d1 * d2 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d x %d] to [%d x %d x %d]\n", dim[0], dim[1], dim[2], dim[3], dim[4], d0, d1, d2);
            assert(d0 * d1 * d2 == dimProd[0] * dim[0]);
        }
        return Array3NonOwn(data(), d0, d1, d2);
    }
    Array3NonOwn<const T> reshape(i64 d0, i64 d1, i64 d2) const
    {
        if (d0 * d1 * d2 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d x %d] to [%d x %d x %d]\n", dim[0], dim[1], dim[2], dim[3], dim[4], d0, d1, d2);
            assert(d0 * d1 * d2 == dimProd[0] * dim[0]);
        }
        return Array3NonOwn(data(), d0, d1, d2);
    }

    Array4NonOwn<T> reshape(i64 d0, i64 d1, i64 d2, i64 d3)
    {
        if (d0 * d1 * d2 * d3 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d x %d] to [%d x %d x %d x %d]\n", dim[0], dim[1], dim[2], dim[3], dim[4], d0, d1, d2, d3);
            assert(d0 * d1 * d2 * d3 == dimProd[0] * dim[0]);
        }
        return Array4NonOwn(data(), d0, d1, d2, d3);
    }
    Array4NonOwn<const T> reshape(i64 d0, i64 d1, i64 d2, i64 d3) const
    {
        if (d0 * d1 * d2 * d3 != dimProd[0] * dim[0])
        {
            printf("Cannot reshape array [%d x %d x %d x %d x %d] to [%d x %d x %d x %d]\n", dim[0], dim[1], dim[2], dim[3], dim[4], d0, d1, d2, d3);
            assert(d0 * d1 * d2 * d3 == dimProd[0] * dim[0]);
        }
        return Array4NonOwn(data(), d0, d1, d2, d3);
    }
};

}

#ifndef CMO_ARRAY_NO_TYPEDEF
template <typename T>
using View = Jasnah::Array1NonOwn<T>;
template <typename T>
using ConstView = Jasnah::Array1NonOwn<const T>;
template <typename T>
using View1D = Jasnah::Array1NonOwn<T>;
template <typename T>
using ConstView1D = Jasnah::Array1NonOwn<const T>;
template <typename T>
using View2D = Jasnah::Array2NonOwn<T>;
template <typename T>
using ConstView2D = Jasnah::Array2NonOwn<const T>;
template <typename T>
using View3D = Jasnah::Array3NonOwn<T>;
template <typename T>
using ConstView3D = Jasnah::Array3NonOwn<const T>;
template <typename T>
using View4D = Jasnah::Array4NonOwn<T>;
template <typename T>
using ConstView4D = Jasnah::Array4NonOwn<const T>;
template <typename T>
using View5D = Jasnah::Array5NonOwn<T>;
template <typename T>
using ConstView5D = Jasnah::Array5NonOwn<const T>;
typedef View<double> F64View;
typedef View<double> F64View1D;
typedef View2D<double> F64View2D;
typedef View3D<double> F64View3D;
typedef View4D<double> F64View4D;
typedef View5D<double> F64View5D;

template <typename T>
using Arr = Jasnah::Array1Own<T>;
template <typename T>
using Arr1D = Jasnah::Array1Own<T>;
template <typename T>
using Arr2D = Jasnah::Array2Own<T>;
template <typename T>
using Arr3D = Jasnah::Array3Own<T>;
template <typename T>
using Arr4D = Jasnah::Array4Own<T>;
template <typename T>
using Arr5D = Jasnah::Array5Own<T>;
typedef Jasnah::Array1Own<double> F64Arr;
typedef Jasnah::Array1Own<double> F64Arr1D;
typedef Jasnah::Array2Own<double> F64Arr2D;
typedef Jasnah::Array3Own<double> F64Arr3D;
typedef Jasnah::Array4Own<double> F64Arr4D;
typedef Jasnah::Array5Own<double> F64Arr5D;
#endif

#else
#endif