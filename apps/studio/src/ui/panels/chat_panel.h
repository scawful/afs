#pragma once

#include "../../models/state.h"
#include "../../core/llama_client.h"
#include <functional>
#include <string>

namespace afs::viz::ui {

// A clean, log-based chat interface with llama.cpp integration.
void RenderChatPanel(AppState& state,
                     LlamaClient& llama_client,
                     std::function<void(const std::string&, const std::string&, const std::string&)> log_callback);

} // namespace afs::viz::ui
