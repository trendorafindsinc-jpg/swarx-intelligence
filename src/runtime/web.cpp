#include "swarx/web.hpp"

#include <array>
#include <cerrno>
#include <cstring>
#include <sstream>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace swarx {

HttpResponse HttpClient::get(const HttpRequest& request) const {
    HttpResponse response;
    if (request.url.empty()) {
        response.error = "HTTP request URL cannot be empty";
        return response;
    }
    if (request.url.rfind("https://", 0) != 0 && request.url.rfind("http://", 0) != 0) {
        response.error = "SWARX HTTP client only permits http:// and https:// URLs";
        return response;
    }

    int out_pipe[2]{};
    if (::pipe(out_pipe) != 0) {
        response.error = std::string("pipe failed: ") + std::strerror(errno);
        return response;
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(out_pipe[0]);
        ::close(out_pipe[1]);
        response.error = std::string("fork failed: ") + std::strerror(errno);
        return response;
    }

    if (pid == 0) {
        ::dup2(out_pipe[1], STDOUT_FILENO);
        ::close(out_pipe[0]);
        ::close(out_pipe[1]);

        std::vector<std::string> args;
        args.emplace_back("curl");
        args.emplace_back("--silent");
        args.emplace_back("--show-error");
        args.emplace_back("--location");
        args.emplace_back("--fail-with-body");
        args.emplace_back("--max-time");
        args.emplace_back(std::to_string(request.timeout_seconds));
        args.emplace_back("--connect-timeout");
        args.emplace_back(std::to_string(request.timeout_seconds));
        args.emplace_back("--max-filesize");
        args.emplace_back(std::to_string(request.max_bytes));
        args.emplace_back("--write-out");
        args.emplace_back("\\nSWARX_HTTP_STATUS:%{http_code}");
        for (const auto& header : request.headers) {
            args.emplace_back("--header");
            args.push_back(header);
        }
        args.emplace_back(request.url);

        std::vector<char*> argv;
        argv.reserve(args.size() + 1);
        for (auto& arg : args) argv.push_back(arg.data());
        argv.push_back(nullptr);
        ::execvp("curl", argv.data());
        _exit(127);
    }

    ::close(out_pipe[1]);
    std::string output;
    std::array<char, 4096> buffer{};
    while (true) {
        const auto n = ::read(out_pipe[0], buffer.data(), buffer.size());
        if (n > 0) {
            output.append(buffer.data(), static_cast<std::size_t>(n));
            if (output.size() > request.max_bytes + 512) {
                ::close(out_pipe[0]);
                ::kill(pid, SIGKILL);
                ::waitpid(pid, nullptr, 0);
                response.error = "HTTP response exceeded SWARX maximum size";
                return response;
            }
        } else if (n == 0) {
            break;
        } else if (errno != EINTR) {
            break;
        }
    }
    ::close(out_pipe[0]);

    int exit_status = 0;
    ::waitpid(pid, &exit_status, 0);

    const std::string marker = "\nSWARX_HTTP_STATUS:";
    const auto marker_pos = output.rfind(marker);
    if (marker_pos == std::string::npos) {
        response.error = WIFEXITED(exit_status) && WEXITSTATUS(exit_status) == 127
            ? "curl executable was not found"
            : "HTTP transport did not return a status code";
        return response;
    }

    const auto status_text = output.substr(marker_pos + marker.size());
    try {
        response.status_code = std::stol(status_text);
    } catch (...) {
        response.error = "invalid HTTP status returned by curl";
        return response;
    }
    response.body = output.substr(0, marker_pos);
    if (!(WIFEXITED(exit_status) && WEXITSTATUS(exit_status) == 0) || !response.ok()) {
        std::ostringstream error;
        error << "HTTP GET failed with status " << response.status_code;
        if (WIFEXITED(exit_status) && WEXITSTATUS(exit_status) != 0) {
            error << " (curl exit " << WEXITSTATUS(exit_status) << ")";
        }
        response.error = error.str();
    }
    return response;
}

} // namespace swarx
