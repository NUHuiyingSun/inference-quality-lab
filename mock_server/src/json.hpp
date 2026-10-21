// Minimal recursive-descent JSON parser: enough to validate request bodies
// and read top-level fields. Not a general-purpose library.
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace json {

struct Value {
  enum class Type { kNull, kBool, kNumber, kString, kArray, kObject };
  Type type = Type::kNull;
  bool boolean = false;
  double num = 0.0;
  std::string str;
  std::vector<Value> arr;
  std::vector<std::pair<std::string, Value>> obj;

  bool is_null() const { return type == Type::kNull; }
  bool is_bool() const { return type == Type::kBool; }
  bool is_number() const { return type == Type::kNumber; }
  bool is_string() const { return type == Type::kString; }
  bool is_array() const { return type == Type::kArray; }
  bool is_object() const { return type == Type::kObject; }

  const Value* get(const std::string& key) const {
    if (!is_object()) return nullptr;
    for (const auto& kv : obj)
      if (kv.first == key) return &kv.second;
    return nullptr;
  }
};

class Parser {
 public:
  explicit Parser(const std::string& s) : s_(s) {}

  bool parse(Value& out, std::string& err) {
    skip_ws();
    if (!parse_value(out, 0)) { err = err_ + " at offset " + std::to_string(pos_); return false; }
    skip_ws();
    if (pos_ != s_.size()) { err = "trailing characters at offset " + std::to_string(pos_); return false; }
    return true;
  }

 private:
  static constexpr int kMaxDepth = 64;
  const std::string& s_;
  size_t pos_ = 0;
  std::string err_;

  bool fail(const char* m) { err_ = m; return false; }
  void skip_ws() {
    while (pos_ < s_.size() && (s_[pos_] == ' ' || s_[pos_] == '\n' || s_[pos_] == '\r' || s_[pos_] == '\t')) ++pos_;
  }
  bool consume(const char* lit) {
    size_t n = std::char_traits<char>::length(lit);
    if (s_.compare(pos_, n, lit) != 0) return false;
    pos_ += n;
    return true;
  }

  bool parse_value(Value& v, int depth) {
    if (depth > kMaxDepth) return fail("nesting too deep");
    if (pos_ >= s_.size()) return fail("unexpected end of input");
    char c = s_[pos_];
    if (c == '{') return parse_object(v, depth);
    if (c == '[') return parse_array(v, depth);
    if (c == '"') { v.type = Value::Type::kString; return parse_string(v.str); }
    if (consume("true")) { v.type = Value::Type::kBool; v.boolean = true; return true; }
    if (consume("false")) { v.type = Value::Type::kBool; v.boolean = false; return true; }
    if (consume("null")) { v.type = Value::Type::kNull; return true; }
    if (c == '-' || (c >= '0' && c <= '9')) return parse_number(v);
    return fail("unexpected character");
  }

  bool parse_number(Value& v) {
    const char* begin = s_.c_str() + pos_;
    char* end = nullptr;
    double d = std::strtod(begin, &end);
    if (end == begin) return fail("invalid number");
    pos_ += static_cast<size_t>(end - begin);
    v.type = Value::Type::kNumber;
    v.num = d;
    return true;
  }

  static void append_utf8(std::string& out, unsigned cp) {
    if (cp < 0x80) out += static_cast<char>(cp);
    else if (cp < 0x800) { out += static_cast<char>(0xC0 | (cp >> 6)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
    else { out += static_cast<char>(0xE0 | (cp >> 12)); out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F)); out += static_cast<char>(0x80 | (cp & 0x3F)); }
  }

  bool parse_string(std::string& out) {
    ++pos_;  // opening quote
    while (pos_ < s_.size()) {
      char c = s_[pos_++];
      if (c == '"') return true;
      if (static_cast<unsigned char>(c) < 0x20) return fail("control character in string");
      if (c != '\\') { out += c; continue; }
      if (pos_ >= s_.size()) break;
      char e = s_[pos_++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          if (pos_ + 4 > s_.size()) return fail("bad unicode escape");
          unsigned cp = static_cast<unsigned>(std::strtoul(s_.substr(pos_, 4).c_str(), nullptr, 16));
          pos_ += 4;
          append_utf8(out, cp);
          break;
        }
        default: return fail("bad escape");
      }
    }
    return fail("unterminated string");
  }

  bool parse_array(Value& v, int depth) {
    v.type = Value::Type::kArray;
    ++pos_;
    skip_ws();
    if (pos_ < s_.size() && s_[pos_] == ']') { ++pos_; return true; }
    while (true) {
      Value item;
      skip_ws();
      if (!parse_value(item, depth + 1)) return false;
      v.arr.push_back(std::move(item));
      skip_ws();
      if (pos_ >= s_.size()) return fail("unterminated array");
      if (s_[pos_] == ',') { ++pos_; continue; }
      if (s_[pos_] == ']') { ++pos_; return true; }
      return fail("expected ',' or ']'");
    }
  }

  bool parse_object(Value& v, int depth) {
    v.type = Value::Type::kObject;
    ++pos_;
    skip_ws();
    if (pos_ < s_.size() && s_[pos_] == '}') { ++pos_; return true; }
    while (true) {
      skip_ws();
      if (pos_ >= s_.size() || s_[pos_] != '"') return fail("expected object key");
      std::string key;
      if (!parse_string(key)) return false;
      skip_ws();
      if (pos_ >= s_.size() || s_[pos_] != ':') return fail("expected ':'");
      ++pos_;
      skip_ws();
      Value item;
      if (!parse_value(item, depth + 1)) return false;
      v.obj.emplace_back(std::move(key), std::move(item));
      skip_ws();
      if (pos_ >= s_.size()) return fail("unterminated object");
      if (s_[pos_] == ',') { ++pos_; continue; }
      if (s_[pos_] == '}') { ++pos_; return true; }
      return fail("expected ',' or '}'");
    }
  }
};

inline bool parse(const std::string& s, Value& out, std::string& err) {
  return Parser(s).parse(out, err);
}

inline std::string quote(const std::string& s) {
  std::string o = "\"";
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (c < 0x20) { char buf[8]; std::snprintf(buf, sizeof(buf), "\\u%04x", c); o += buf; }
        else o += static_cast<char>(c);
    }
  }
  return o + "\"";
}

}  // namespace json
