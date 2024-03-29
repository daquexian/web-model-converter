#pragma once

#include <cstdint>
#include <cstdio>
#include <vector>
#include <string>

class FakeFile {
    private:
        FILE *fp = nullptr;
        char *buf = nullptr;
        size_t size = 0;
    public:
        FakeFile() {
            fp = open_memstream (&buf, &size);
        }
        void Open() {
            if (!fp) {
                fp = open_memstream (&buf, &size);
            }
        }
        operator FILE* () {
            return fp;
        }
        std::string CloseAndGetStr() {
            if (fp) {
                fclose(fp);
                fp = nullptr;
            }
            return std::string(buf, size);
        }
        std::pair<void *, size_t> CloseAndGetBuf() {
            if (fp) {
                fclose(fp);
                fp = nullptr;
            }
            return std::make_pair(buf, size);
        }
        FakeFile &operator=(FakeFile) = delete;
        FakeFile(const FakeFile &) = delete;
        ~FakeFile() {
            if (fp) {
                fclose(fp);
                fp = nullptr;
            }
        }
};

inline std::string ReplaceAll(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
    return str;
}


// param, bin, error msg
using Buffer = std::pair<void *, size_t>;
using NcnnModel = std::tuple<Buffer, Buffer, std::string>;
