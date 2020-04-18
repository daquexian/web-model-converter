#pragma once

#include <cstdint>
#include <cstdio>
#include <vector>
#include <string>

using FakeFile = std::vector<char>;

inline void fwrite(const void *ptr, size_t size, size_t count, FakeFile &vec) {
    const auto *char_ptr = reinterpret_cast<const char *>(ptr);

    for (size_t i = 0; i < size * count; i++) {
        vec.push_back(char_ptr[i]);
    }
}

template <typename... Args>
void fprintf(FakeFile &vec, const char *format, Args... args) {
    char buffer[999];
    const auto n = sprintf(buffer, format, args...);
    for (int i = 0; i < n; i++) {
        vec.push_back(buffer[i]);
    }
}

inline long int ftell(FakeFile &vec) {
    return vec.size();
}

inline std::string ReplaceAll(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
    return str;
}


// param, bin, error msg
using NcnnModel = std::tuple<std::vector<char>, std::vector<char>, std::string>;
