#pragma once

#include <cstdint>
#include <cstdio>
#include <vector>
#include <string>

using Buffer = std::pair<void *, size_t>;
// param, bin, error msg
using NcnnModel = std::tuple<Buffer, Buffer, std::string>;

struct WasmBuffer {
  unsigned char *output_buffer1 = nullptr;
  unsigned char *output_buffer2 = nullptr;
  unsigned char *output_buffer3 = nullptr;
  size_t output_buffer_size1 = 0;
  size_t output_buffer_size2 = 0;
  size_t output_buffer_size3 = 0;

  void freeBuffers() {
    freeBuffer1();
    freeBuffer2();
    freeBuffer3();
  }
  void freeBuffer1() {
    if (output_buffer1 != nullptr) {
      free(output_buffer1);
      output_buffer1 = nullptr;
      output_buffer_size1 = 0;
    }
  }
  void freeBuffer2() {
    if (output_buffer2 != nullptr) {
      free(output_buffer2);
      output_buffer2 = nullptr;
      output_buffer_size2 = 0;
    }
  }
  void freeBuffer3() {
    if (output_buffer3 != nullptr) {
      free(output_buffer3);
      output_buffer3 = nullptr;
      output_buffer_size3 = 0;
    }
  }
  void setBuffer1(Buffer buf) { setBuffer1(buf.first, buf.second); }
  void setBuffer1(void *buf, const size_t buflen) {
    // we own the buf
    output_buffer1 = static_cast<unsigned char *>(buf);
    output_buffer_size1 = buflen;
  }
  void setBuffer1(const std::string &str) {
    output_buffer1 = static_cast<unsigned char *>(malloc(str.size()));
    memcpy(output_buffer1, str.c_str(), str.size());
    output_buffer_size1 = str.size();
  }
  void setBuffer2(Buffer buf) { setBuffer2(buf.first, buf.second); }
  void setBuffer2(void *buf, const size_t buflen) {
    // we own the buf
    output_buffer2 = static_cast<unsigned char *>(buf);
    output_buffer_size2 = buflen;
  }
  void setBuffer2(const std::string &str) {
    output_buffer2 = static_cast<unsigned char *>(malloc(str.size()));
    memcpy(output_buffer2, str.c_str(), str.size());
    output_buffer_size2 = str.size();
  }
  void setBuffer3(const std::string &str) {
    output_buffer3 = static_cast<unsigned char *>(malloc(str.size()));
    memcpy(output_buffer3, str.c_str(), str.size());
    output_buffer_size3 = str.size();
  }
};

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

