#pragma once

#include <array>
#include <type_traits>

template <typename Dest=void, typename ...Arg>
constexpr auto make_array(Arg&& ...arg) {
   if constexpr (std::is_same<void,Dest>::value)
      return std::array<std::common_type_t<std::decay_t<Arg>...>, sizeof...(Arg)>{{ std::forward<Arg>(arg)... }};
   else
      return std::array<Dest, sizeof...(Arg)>{{ std::forward<Arg>(arg)... }};
}
