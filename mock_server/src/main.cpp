// OpenAI-compatible mock inference server with fault injection.
//
// Endpoints:
//   GET  /health
//   POST /v1/chat/completions   (stream: true|false)
//
// Fault injection via request header `X-Mock-Fault` (comma-separated):
//   status=429|500|503     respond with that error status (429 adds Retry-After)
//   retry_after=N          Retry-After seconds for 429 (default 1)
//   latency_ms=N           sleep before sending any byte
//   chunk_delay_ms=N       sleep between SSE chunks
//   drop_after=N           close the connection after N content chunks (no [DONE])
//   malformed_chunk=N      emit an invalid JSON payload as content chunk N
//   no_done                omit the terminating `data: [DONE]`
//
// Auth: if env MOCK_API_KEY is set, requests need `Authorization: Bearer <key>`.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "json.hpp"

// macOS has no MSG_NOSIGNAL; SIGPIPE is ignored in main(), so 0 is safe there.
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

namespace {

std::atomic<long> g_request_counter{0};
std::mutex g_log_mutex;
constexpr size_t kMaxBody = 1 << 20;

const std::vector<std::string> kWords = {
    "Fast", "inference", "lets", "applications", "respond", "in",
    "real", "time", "and", "iterate", "more", "often."};

void log_line(const std::string& s) {
  std::lock_guard<std::mutex> lock(g_log_mutex);
  std::fprintf(stderr, "%s\n", s.c_str());
}

bool send_all(int fd, const std::string& data) {
  size_t sent = 0;
  while (sent < data.size()) {
    ssize_t n = ::send(fd, data.data() + sent, data.size() - sent, MSG_NOSIGNAL);
    if (n <= 0) return false;
    sent += static_cast<size_t>(n);
  }
  return true;
}

void sleep_ms(long ms) {
  if (ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

std::string to_lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}

std::string trim(const std::string& s) {
  size_t b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return "";
  size_t e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

struct Request {
  std::string method;
  std::string path;
  std::map<std::string, std::string> headers;  // lower-cased keys
  std::string body;
};

enum class ReadResult { kOk, kClosed, kBadRequest, kTooLarge };

ReadResult read_request(int fd, Request& req) {
  std::string buf;
  char tmp[4096];
  size_t header_end = std::string::npos;
  while ((header_end = buf.find("\r\n\r\n")) == std::string::npos) {
    ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
    if (n <= 0) return ReadResult::kClosed;
    buf.append(tmp, static_cast<size_t>(n));
    if (buf.size() > 64 * 1024) return ReadResult::kTooLarge;
  }
  std::istringstream hs(buf.substr(0, header_end));
  std::string line;
  if (!std::getline(hs, line)) return ReadResult::kBadRequest;
  {
    std::istringstream rl(trim(line));
    std::string version;
    if (!(rl >> req.method >> req.path >> version)) return ReadResult::kBadRequest;
  }
  while (std::getline(hs, line)) {
    auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    req.headers[to_lower(trim(line.substr(0, colon)))] = trim(line.substr(colon + 1));
  }
  size_t content_length = 0;
  auto it = req.headers.find("content-length");
  if (it != req.headers.end()) {
    char* end = nullptr;
    unsigned long v = std::strtoul(it->second.c_str(), &end, 10);
    if (end == it->second.c_str()) return ReadResult::kBadRequest;
    content_length = v;
  }
  if (content_length > kMaxBody) return ReadResult::kTooLarge;
  req.body = buf.substr(header_end + 4);
  while (req.body.size() < content_length) {
    ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
    if (n <= 0) return ReadResult::kClosed;
    req.body.append(tmp, static_cast<size_t>(n));
  }
  req.body.resize(content_length);
  return ReadResult::kOk;
}

const char* reason_phrase(int status) {
  switch (status) {
    case 200: return "OK";
    case 400: return "Bad Request";
    case 401: return "Unauthorized";
    case 404: return "Not Found";
    case 405: return "Method Not Allowed";
    case 413: return "Payload Too Large";
    case 429: return "Too Many Requests";
    case 500: return "Internal Server Error";
    case 503: return "Service Unavailable";
    default: return "Error";
  }
}

void send_json(int fd, int status, const std::string& body,
               const std::vector<std::string>& extra_headers = {}) {
  std::ostringstream os;
  os << "HTTP/1.1 " << status << ' ' << reason_phrase(status) << "\r\n"
     << "Content-Type: application/json\r\n"
     << "Content-Length: " << body.size() << "\r\n"
     << "Connection: close\r\n";
  for (const auto& h : extra_headers) os << h << "\r\n";
  os << "\r\n" << body;
  send_all(fd, os.str());
}

void send_error(int fd, int status, const std::string& type, const std::string& message,
                const std::string& param = "",
                const std::vector<std::string>& extra_headers = {}) {
  std::string body = "{\"error\":{\"message\":" + json::quote(message) +
                     ",\"type\":" + json::quote(type) + ",\"param\":" +
                     (param.empty() ? std::string("null") : json::quote(param)) +
                     ",\"code\":" + json::quote(type) + "}}";
  send_json(fd, status, body, extra_headers);
}

struct Faults {
  int status = 0;
  long retry_after = 1;
  long latency_ms = 0;
  long chunk_delay_ms = 0;
  long drop_after = -1;
  long malformed_chunk = -1;
  bool no_done = false;
};

Faults parse_faults(const std::map<std::string, std::string>& headers) {
  Faults f;
  auto it = headers.find("x-mock-fault");
  if (it == headers.end()) return f;
  std::istringstream ss(it->second);
  std::string item;
  while (std::getline(ss, item, ',')) {
    item = trim(item);
    auto eq = item.find('=');
    std::string key = eq == std::string::npos ? item : item.substr(0, eq);
    long val = eq == std::string::npos ? 0 : std::atol(item.substr(eq + 1).c_str());
    if (key == "status") f.status = static_cast<int>(val);
    else if (key == "retry_after") f.retry_after = val;
    else if (key == "latency_ms") f.latency_ms = val;
    else if (key == "chunk_delay_ms") f.chunk_delay_ms = val;
    else if (key == "drop_after") f.drop_after = val;
    else if (key == "malformed_chunk") f.malformed_chunk = val;
    else if (key == "no_done") f.no_done = true;
  }
  return f;
}

// Returns an empty string when valid, otherwise "param|message".
std::string validate_chat_request(const json::Value& root) {
  if (!root.is_object()) return "|Request body must be a JSON object.";
  const json::Value* model = root.get("model");
  if (!model || !model->is_string() || model->str.empty())
    return "model|'model' is required and must be a non-empty string.";
  const json::Value* messages = root.get("messages");
  if (!messages || !messages->is_array())
    return "messages|'messages' is required and must be an array.";
  if (messages->arr.empty()) return "messages|'messages' must contain at least one message.";
  static const std::vector<std::string> roles = {"system", "user", "assistant", "tool"};
  for (size_t i = 0; i < messages->arr.size(); ++i) {
    const json::Value& m = messages->arr[i];
    std::string where = "messages[" + std::to_string(i) + "]";
    if (!m.is_object()) return where + "|Each message must be an object.";
    const json::Value* role = m.get("role");
    if (!role || !role->is_string() ||
        std::find(roles.begin(), roles.end(), role->str) == roles.end())
      return where + ".role|Invalid or missing role.";
    const json::Value* content = m.get("content");
    bool ok = content && (content->is_string() ||
                          (content->is_null() && role->str == "assistant"));
    if (!ok) return where + ".content|'content' must be a string.";
  }
  if (const json::Value* t = root.get("temperature")) {
    if (!t->is_number() || t->num < 0.0 || t->num > 2.0)
      return "temperature|'temperature' must be a number between 0 and 2.";
  }
  if (const json::Value* mt = root.get("max_tokens")) {
    if (!mt->is_number() || mt->num < 1 || std::floor(mt->num) != mt->num)
      return "max_tokens|'max_tokens' must be a positive integer.";
  }
  if (const json::Value* s = root.get("stream")) {
    if (!s->is_bool()) return "stream|'stream' must be a boolean.";
  }
  return "";
}

long count_prompt_tokens(const json::Value& root) {
  long n = 0;
  for (const auto& m : root.get("messages")->arr) {
    const json::Value* c = m.get("content");
    if (!c || !c->is_string()) continue;
    std::istringstream ws(c->str);
    std::string w;
    while (ws >> w) ++n;
  }
  return n;
}

std::string usage_json(long prompt, long completion) {
  return "{\"prompt_tokens\":" + std::to_string(prompt) + ",\"completion_tokens\":" +
         std::to_string(completion) + ",\"total_tokens\":" + std::to_string(prompt + completion) +
         "}";
}

std::string chunk_json(const std::string& id, long created, const std::string& model,
                       const std::string& delta, const std::string& finish,
                       const std::string& usage) {
  std::string s = "{\"id\":" + json::quote(id) +
                  ",\"object\":\"chat.completion.chunk\",\"created\":" + std::to_string(created) +
                  ",\"model\":" + json::quote(model) + ",\"choices\":[{\"index\":0,\"delta\":" +
                  delta + ",\"finish_reason\":" + (finish.empty() ? "null" : json::quote(finish)) +
                  "}]";
  if (!usage.empty()) s += ",\"usage\":" + usage;
  return s + "}";
}

// Optional `X-Mock-Echo: <marker>` makes the first output token "[marker]" so concurrent
// tests can verify each stream received its own content (no cross-request leakage).
std::string echo_marker(const std::map<std::string, std::string>& headers) {
  auto it = headers.find("x-mock-echo");
  if (it == headers.end()) return "";
  std::string m;
  for (char c : it->second)
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_') m += c;
  return m.substr(0, 64);
}

void handle_chat(int fd, const Request& req, const Faults& faults) {
  json::Value root;
  std::string parse_err;
  if (!json::parse(req.body, root, parse_err)) {
    send_error(fd, 400, "invalid_request_error", "Invalid JSON body: " + parse_err);
    return;
  }
  std::string verr = validate_chat_request(root);
  if (!verr.empty()) {
    auto bar = verr.find('|');
    send_error(fd, 400, "invalid_request_error", verr.substr(bar + 1), verr.substr(0, bar));
    return;
  }

  const std::string model = root.get("model")->str;
  const json::Value* mt = root.get("max_tokens");
  const long natural = static_cast<long>(kWords.size());
  const long limit = mt ? static_cast<long>(mt->num) : natural;
  const long n_out = std::min(limit, natural);
  const std::string finish = (limit < natural) ? "length" : "stop";
  const long prompt_tokens = count_prompt_tokens(root);
  const long created = static_cast<long>(std::time(nullptr));
  const std::string id = "chatcmpl-mock-" + std::to_string(++g_request_counter);
  const json::Value* stream = root.get("stream");
  const bool streaming = stream && stream->boolean;

  std::vector<std::string> pieces;
  const std::string marker = echo_marker(req.headers);
  for (long i = 0; i < n_out; ++i) {
    const std::string& w = (i == 0 && !marker.empty()) ? "[" + marker + "]" : kWords[i];
    pieces.push_back((i == 0 ? "" : " ") + w);
  }

  if (!streaming) {
    std::string content;
    for (const auto& p : pieces) content += p;
    std::string body = "{\"id\":" + json::quote(id) +
                       ",\"object\":\"chat.completion\",\"created\":" + std::to_string(created) +
                       ",\"model\":" + json::quote(model) +
                       ",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
                       "\"content\":" + json::quote(content) + "},\"finish_reason\":" +
                       json::quote(finish) + "}],\"usage\":" + usage_json(prompt_tokens, n_out) +
                       "}";
    send_json(fd, 200, body);
    return;
  }

  if (!send_all(fd, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                    "Cache-Control: no-cache\r\nConnection: close\r\n\r\n"))
    return;
  auto emit = [&](const std::string& payload) {
    sleep_ms(faults.chunk_delay_ms);
    return send_all(fd, "data: " + payload + "\n\n");
  };
  if (!emit(chunk_json(id, created, model, "{\"role\":\"assistant\",\"content\":\"\"}", "", "")))
    return;
  for (long i = 0; i < n_out; ++i) {
    if (faults.drop_after >= 0 && i >= faults.drop_after) return;  // abrupt close
    std::string payload = (faults.malformed_chunk == i)
                              ? std::string("{\"id\":\"") + id + "\",\"choices\":[{\"delta\":"
                              : chunk_json(id, created, model,
                                           "{\"content\":" + json::quote(pieces[i]) + "}", "", "");
    if (!emit(payload)) return;
  }
  if (!emit(chunk_json(id, created, model, "{}", finish, usage_json(prompt_tokens, n_out)))) return;
  if (!faults.no_done) emit("[DONE]");
}

void handle_connection(int fd, const std::string& api_key) {
  Request req;
  ReadResult rr = read_request(fd, req);
  if (rr == ReadResult::kClosed) { ::close(fd); return; }
  if (rr == ReadResult::kTooLarge) {
    send_error(fd, 413, "invalid_request_error", "Request too large.");
    ::close(fd);
    return;
  }
  if (rr == ReadResult::kBadRequest) {
    send_error(fd, 400, "invalid_request_error", "Malformed HTTP request.");
    ::close(fd);
    return;
  }
  auto t0 = std::chrono::steady_clock::now();
  Faults faults = parse_faults(req.headers);
  sleep_ms(faults.latency_ms);

  if (req.path == "/health") {
    send_json(fd, 200, "{\"status\":\"ok\"}");
  } else if (req.path != "/v1/chat/completions") {
    send_error(fd, 404, "not_found_error", "Unknown route: " + req.path);
  } else if (req.method != "POST") {
    send_error(fd, 405, "invalid_request_error", "Use POST.");
  } else if (!api_key.empty() && req.headers["authorization"] != "Bearer " + api_key) {
    send_error(fd, 401, "authentication_error", "Invalid or missing API key.");
  } else if (faults.status == 429) {
    send_error(fd, 429, "rate_limit_error", "Rate limit exceeded (injected).", "",
               {"Retry-After: " + std::to_string(faults.retry_after)});
  } else if (faults.status >= 500) {
    send_error(fd, faults.status, "server_error", "Injected server error.");
  } else {
    handle_chat(fd, req, faults);
  }
  long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();
  log_line(req.method + " " + req.path + " " + std::to_string(ms) + "ms");
  ::close(fd);
}

}  // namespace

int main(int argc, char** argv) {
  std::signal(SIGPIPE, SIG_IGN);
  int port = argc > 1 ? std::atoi(argv[1]) : 8080;
  const char* key_env = std::getenv("MOCK_API_KEY");
  std::string api_key = key_env ? key_env : "";

  int server = ::socket(AF_INET, SOCK_STREAM, 0);
  if (server < 0) { std::perror("socket"); return 1; }
  int yes = 1;
  ::setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port = htons(static_cast<uint16_t>(port));
  if (::bind(server, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    std::perror("bind");
    return 1;
  }
  if (::listen(server, 128) < 0) { std::perror("listen"); return 1; }
  log_line("mock server listening on 127.0.0.1:" + std::to_string(port) +
           (api_key.empty() ? " (auth disabled)" : " (auth enabled)"));

  while (true) {
    int client = ::accept(server, nullptr, nullptr);
    if (client < 0) continue;
    std::thread(handle_connection, client, api_key).detach();
  }
}
