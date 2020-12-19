// Copyright 2019 JD.com Inc. JD AI

#ifndef DNN_LOG_HELPER_H
#define DNN_LOG_HELPER_H

#include <array>
#include <iostream>
#include <vector>

template <typename T, size_t N>
std::ostream &operator<<(std::ostream &output, std::array<T, N> const &values);

template <typename T>
std::ostream &operator<<(std::ostream &output, std::vector<T> const &values) {
  output << "[";
  for (size_t i = 0; i < values.size(); i++) {
    output << values[i];
    if (i != values.size() - 1) {
      output << ", ";
    }
  }
  output << "]";
  return output;
}

template <typename T, size_t N>
std::ostream &operator<<(std::ostream &output, std::array<T, N> const &values) {
  output << "[";
  for (size_t i = 0; i < values.size(); i++) {
    output << values[i];
    if (i != values.size() - 1) {
      output << ", ";
    }
  }
  output << "]";
  return output;
}

// Make a FOREACH macro
#define FE_1(WHAT, X) WHAT(X)
#define FE_2(WHAT, X, ...) WHAT(X) FE_1(WHAT, __VA_ARGS__)
#define FE_3(WHAT, X, ...) WHAT(X) FE_2(WHAT, __VA_ARGS__)
#define FE_4(WHAT, X, ...) WHAT(X) FE_3(WHAT, __VA_ARGS__)
#define FE_5(WHAT, X, ...) WHAT(X) FE_4(WHAT, __VA_ARGS__)
#define FE_6(WHAT, X, ...) WHAT(X) FE_5(WHAT, __VA_ARGS__)
#define FE_7(WHAT, X, ...) WHAT(X) FE_6(WHAT, __VA_ARGS__)
#define FE_8(WHAT, X, ...) WHAT(X) FE_7(WHAT, __VA_ARGS__)
//... repeat as needed

#define DQX_GET_MACRO(_1, _2, _3, _4, _5, _6, _7, _8, NAME, ...) NAME
#define FOR_EACH(action, ...)                                              \
    DQX_GET_MACRO(__VA_ARGS__, FE_8, FE_7, FE_6, FE_5, FE_4, FE_3, FE_2, FE_1) \
    (action, __VA_ARGS__)

#define M_A_1(_1, ...) _1
#define M_A_2(_1, _2, ...) _2
#define M_A_3(_1, _2, _3, ...) _3
#define M_A_4(_1, _2, _3, _4, ...) _4
#define M_A_5(_1, _2, _3, _4, _5, ...) _5
#define M_A_6(_1, _2, _3, _4, _5, _6, ...) _6
#define M_A_7(_1, _2, _3, _4, _5, _6, _7, ...) _7
#define M_A_8(_1, _2, _3, _4, _5, _6, _7, _8, ...) _8

#define FIRST_ARG(...) M_A_1(__VA_ARGS__)

#define LAST_ARG(...)                                                       \
    DQX_GET_MACRO(__VA_ARGS__, M_A_8, M_A_7, M_A_6, M_A_5, M_A_4, M_A_3, M_A_2, \
              M_A_1)                                                        \
    (__VA_ARGS__)

#define FORZS(var, end, step) \
    for (auto var = decltype(end){0}; var < end; var += (step))

#define FORZ(var, end) for (auto var = decltype(end){0}; var < end; var++)

#define FOR(var, start, end) \
    for (auto var = decltype(end){start}; var < end; var++)

#define STR(a) #a
#define XSTR(a) STR(a)

#define PNT_STR(s) << s << " "
#define PNT_VAR(var) << XSTR(var) << " = " << (var) << ", "
#define PNT_TO(stream, ...) stream FOR_EACH(PNT_VAR, __VA_ARGS__)
#define PNT(...) PNT_TO(std::cout, __VA_ARGS__) << std::endl;

// C++ version of Python map start
namespace adl_helper {
using std::begin;
using std::end;

template <typename C>
auto adl_begin(C &&c) -> decltype(begin(std::forward<C>(c))) {
  return begin(std::forward<C>(c));
}

template <typename C>
auto adl_end(C &&c) -> decltype(end(std::forward<C>(c))) {
  return end(std::forward<C>(c));
}
}  // namespace adl_helper

using adl_helper::adl_begin;
using adl_helper::adl_end;

template <class T>
using dqx_decay_t = typename std::decay<T>::type;

// apply is C++ version of Python `map`
template <typename C, typename F,
          typename E = dqx_decay_t<
              decltype(std::declval<F>()(*adl_begin(std::declval<C>())))>>
std::vector<E> Apply(C &&container, F &&func) {
  /* mostly same as before, except using adl_begin/end instead
     of unqualified begin/end with using
  */

  std::vector<E> result;
  auto first = adl_begin(std::forward<C>(container));
  auto last = adl_end(std::forward<C>(container));

  result.reserve(std::distance(first, last));
  for (; first != last; ++first) {
    result.push_back(std::forward<F>(func)(*first));
  }
  return result;
}
// C++ version of Python map end

#endif

