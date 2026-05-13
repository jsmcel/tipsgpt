const els = {
  app: document.querySelector(".app"),
  askForm: document.getElementById("askForm"),
  askBtn: document.getElementById("askBtn"),
  stopBtn: document.getElementById("stopBtn"),
  clearChatBtn: document.getElementById("clearChatBtn"),
  newChatBtn: document.getElementById("newChatBtn"),
  sidebarBtn: document.getElementById("sidebarBtn"),
  refsBtn: document.getElementById("refsBtn"),
  closeRefsBtn: document.getElementById("closeRefsBtn"),
  exportBtn: document.getElementById("exportBtn"),
  question: document.getElementById("question"),
  messages: document.getElementById("messages"),
  refs: document.getElementById("refs"),
  refsSummary: document.getElementById("refsSummary"),
  jumpBottomBtn: document.getElementById("jumpBottomBtn"),
  chatList: document.getElementById("chatList"),
  chatSearch: document.getElementById("chatSearch"),
  chatTitle: document.getElementById("chatTitle"),
  chatMain: document.querySelector(".chat-main"),
  engineBadge: document.getElementById("engineBadge"),
  indexBadge: document.getElementById("indexBadge"),
  docCount: document.getElementById("docCount"),
  chunkCount: document.getElementById("chunkCount"),
  topK: document.getElementById("topK"),
  modelPreset: document.getElementById("modelPreset"),
  modelPresetMobile: document.getElementById("modelPresetMobile"),
  quickPrompts: document.getElementById("quickPrompts"),
  toast: document.getElementById("toast"),
};

const STORE_KEY = "tips-premium-chat-v2";
const LEGACY_KEY = "tips-local-chat";

let state = {
  chats: [],
  activeId: null,
  mode: "answer",
  modelPreset: "codex_high",
  busy: false,
  aborter: null,
  activeJobId: null,
  lastRefs: [],
  scrollIntent: "bottom",
};

const MODEL_LABELS = {
  codex_high: "Codex High",
  codex_fast: "Codex Fast",
  local_rag: "Local RAG",
};
const ASK_START_TIMEOUT_MS = 20000;
const ASK_POLL_TIMEOUT_MS = 12000;
const ASK_POLL_INTERVAL_MS = 1200;
const ASK_POLL_MAX_MISSES = 8;

function uid() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

function nowIso() {
  return new Date().toISOString();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function activeChat() {
  return state.chats.find((chat) => chat.id === state.activeId) || null;
}

function titleFromQuestion(text) {
  const clean = String(text || "").replace(/\s+/g, " ").trim();
  return clean.length > 54 ? `${clean.slice(0, 51)}...` : clean || "New chat";
}

function createChat(title = "New chat") {
  const chat = {
    id: uid(),
    title,
    createdAt: nowIso(),
    updatedAt: nowIso(),
    messages: [],
    refs: [],
    engine: "Codex High",
  };
  state.chats.unshift(chat);
  state.activeId = chat.id;
  saveState();
  renderAll();
  return chat;
}

function loadState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
    if (saved?.chats?.length) {
      state.chats = saved.chats;
      state.activeId = saved.activeId || saved.chats[0].id;
      return;
    }
  } catch {
    // ignore corrupt storage
  }

  try {
    const legacy = JSON.parse(localStorage.getItem(LEGACY_KEY) || "[]");
    if (legacy.length) {
      state.chats = [{
        id: uid(),
        title: "Imported chat",
        createdAt: nowIso(),
        updatedAt: nowIso(),
        messages: legacy,
        refs: [],
        engine: "Codex High",
      }];
      state.activeId = state.chats[0].id;
      saveState();
      return;
    }
  } catch {
    // ignore legacy import errors
  }

  createChat();
}

function saveState() {
  const compact = {
    activeId: state.activeId,
    chats: state.chats.slice(0, 50).map((chat) => ({
      ...chat,
      messages: chat.messages.slice(-80),
      refs: (chat.refs || []).slice(0, 30),
    })),
  };
  localStorage.setItem(STORE_KEY, JSON.stringify(compact));
}

function toast(text) {
  els.toast.textContent = text;
  els.toast.classList.add("show");
  window.setTimeout(() => els.toast.classList.remove("show"), 1600);
}

function abortError() {
  return new DOMException("Operation cancelled", "AbortError");
}

function sleep(ms, signal) {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortError());
      return;
    }
    const timer = window.setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      window.clearTimeout(timer);
      reject(abortError());
    }, { once: true });
  });
}

async function fetchJson(url, options = {}) {
  const {
    timeout = 15000,
    signal,
    retries = 0,
    retryDelay = 500,
    ...fetchOptions
  } = options;

  for (let attempt = 0; attempt <= retries; attempt += 1) {
    const controller = new AbortController();
    const onAbort = () => controller.abort(signal.reason || abortError());
    const timer = window.setTimeout(() => controller.abort(new Error("timeout")), timeout);
    if (signal) {
      if (signal.aborted) throw abortError();
      signal.addEventListener("abort", onAbort, { once: true });
    }

    try {
      const res = await fetch(url, { ...fetchOptions, signal: controller.signal });
      const text = await res.text();
      let data = null;
      if (text) {
        try {
          data = JSON.parse(text);
        } catch {
          data = { detail: text.slice(0, 500) };
        }
      }
      if (!res.ok) {
        throw new Error(data?.detail || `HTTP ${res.status}`);
      }
      return data ?? {};
    } catch (err) {
      if (err.name === "AbortError" || signal?.aborted) throw abortError();
      if (attempt >= retries) throw err;
      await sleep(retryDelay * (attempt + 1), signal);
    } finally {
      window.clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    }
  }
  throw new Error("Could not connect to the server");
}

async function loadManifest() {
  try {
    const data = await fetchJson("/manifest", { timeout: 8000, retries: 1 });
    els.docCount.textContent = data.documents_total ?? data.documents_downloaded ?? "...";
    els.chunkCount.textContent = data.chunks ?? "...";
    els.indexBadge.textContent = data.index_flavour?.includes("bm25") ? "Hybrid BM25" : "Local RAG";
  } catch {
    els.docCount.textContent = "-";
    els.chunkCount.textContent = "-";
    els.indexBadge.textContent = "No index";
  }
}

function renderAll() {
  renderChats();
  renderMessages();
  renderRefs(activeChat()?.refs || []);
  updateTitle();
}

function updateTitle() {
  const chat = activeChat();
  els.chatTitle.textContent = chat?.title || "New chat";
  els.engineBadge.textContent = chat?.engine || currentEngineLabel();
}

function renderChats() {
  const query = els.chatSearch.value.trim().toLowerCase();
  const chats = state.chats.filter((chat) => {
    if (!query) return true;
    const haystack = `${chat.title} ${chat.messages.map((m) => m.content).join(" ")}`.toLowerCase();
    return haystack.includes(query);
  });

  els.chatList.innerHTML = "";
  for (const chat of chats) {
    const last = [...chat.messages].reverse().find((msg) => msg.role !== "system");
    const item = document.createElement("button");
    item.type = "button";
    item.className = `chat-item ${chat.id === state.activeId ? "active" : ""}`;
    item.dataset.chatId = chat.id;
    item.innerHTML = `
      <span class="chat-item-title">${escapeHtml(chat.title)}</span>
      <span class="chat-item-snippet">${escapeHtml(last?.content || "No messages")}</span>
    `;
    els.chatList.appendChild(item);
  }
}

function renderMessages() {
  const chat = activeChat();
  els.messages.innerHTML = "";

  if (!chat || chat.messages.length === 0) {
    els.messages.innerHTML = `
      <div class="welcome">
        <div class="welcome-mark">TIPS</div>
        <h2>TIPS GPT</h2>
        <p>Evidence-first answers with citations and conversational context.</p>
      </div>
    `;
    return;
  }

  for (const [index, msg] of chat.messages.entries()) {
    const node = document.createElement("article");
    node.className = `message ${msg.role} ${msg.pending ? "pending" : ""}`;
    node.dataset.index = index;
    const role = msg.role === "user" ? "You" : "TIPS";
    const actions = msg.pending
      ? `<span class="thinking"><i></i><i></i><i></i></span>`
      : `
        <button class="message-action" type="button" data-copy="${index}">Copy</button>
        ${msg.role === "assistant" ? `<button class="message-action" type="button" data-regenerate="${index}">Regenerate</button>` : ""}
      `;

    node.innerHTML = `
      <div class="avatar">${role.slice(0, 1)}</div>
      <div class="message-content">
        <div class="message-meta">
          <strong>${role}</strong>
          <div class="message-actions">${actions}</div>
        </div>
        <div class="message-body">${renderRichText(msg.content)}</div>
      </div>
    `;
    els.messages.appendChild(node);
  }

  requestAnimationFrame(applyScrollIntent);
}

function applyScrollIntent() {
  if (state.scrollIntent === "assistant-start") {
    const assistants = els.messages.querySelectorAll(".message.assistant");
    const lastAssistant = assistants[assistants.length - 1];
    if (lastAssistant) {
      lastAssistant.scrollIntoView({ block: "start" });
    }
  } else if (state.scrollIntent === "bottom") {
    els.messages.scrollTop = els.messages.scrollHeight;
  }
  state.scrollIntent = "preserve";
  updateJumpButton();
}

function renderRichText(text) {
  let html = escapeHtml(text || "");
  html = html.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${code.trim()}</code></pre>`);
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\[(\d+)\]/g, '<button class="citation-token" type="button" data-citation="$1">[$1]</button>');
  html = html.replace(/\n/g, "<br />");
  return html;
}

function renderRefs(items = []) {
  state.lastRefs = items;
  els.refs.innerHTML = "";
  els.refsSummary.textContent = items.length ? `${items.length} references` : "No references";

  if (!items.length) {
    els.refs.innerHTML = `<div class="refs-empty">Sources for the next answer will appear here.</div>`;
    return;
  }

  for (const item of items) {
    const n = item.n ?? item.ref ?? "?";
    const label = item.label ?? item.citation ?? item.title ?? "Reference";
    const path = item.local_path ?? "";
    const excerpt = item.excerpt ?? "";
    const source = item.source_url ?? "";
    const node = document.createElement("article");
    node.className = "ref";
    node.id = `ref-${n}`;
    node.innerHTML = `
      <div class="ref-top">
        <span>[${escapeHtml(n)}]</span>
        <strong>${escapeHtml(label)}</strong>
      </div>
      ${path ? `<button class="path" type="button" data-copy-path="${escapeHtml(path)}">${escapeHtml(path)}</button>` : ""}
      ${excerpt ? `<details><summary>Excerpt</summary><p>${escapeHtml(excerpt)}</p></details>` : ""}
      ${source ? `<a href="${escapeHtml(source)}" target="_blank" rel="noreferrer">Open ECB source</a>` : ""}
    `;
    els.refs.appendChild(node);
  }
}

function setBusy(value) {
  state.busy = value;
  els.askBtn.disabled = value;
  els.question.disabled = value;
  els.stopBtn.classList.toggle("hidden", !value);
  els.askBtn.textContent = value ? "Thinking" : "Send";
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".mode").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });
  els.engineBadge.textContent = currentEngineLabel();
}

function currentModel() {
  return state.modelPreset || "codex_high";
}

function currentEngineLabel() {
  if (state.mode === "context") return "Local context";
  return MODEL_LABELS[currentModel()] || "Codex High";
}

function setModelPreset(value) {
  const next = MODEL_LABELS[value] ? value : "codex_high";
  state.modelPreset = next;
  if (els.modelPreset && els.modelPreset.value !== next) els.modelPreset.value = next;
  if (els.modelPresetMobile && els.modelPresetMobile.value !== next) els.modelPresetMobile.value = next;
  els.engineBadge.textContent = currentEngineLabel();
}

function visibleHistory(chat) {
  return (chat?.messages || [])
    .filter((item) => item.role === "user" || item.role === "assistant")
    .filter((item) => !item.pending)
    .slice(-10)
    .map((item) => ({ role: item.role, content: item.content }));
}

function progressText(status, elapsedSeconds, misses = 0) {
  if (misses > 0) {
    return `The answer is still running. Retrying the server connection (${misses}/${ASK_POLL_MAX_MISSES})...`;
  }
  if (status === "queued") return "Query queued...";
  if (elapsedSeconds > 90) return "Codex is still working with the retrieved context. Long questions can take a little while...";
  if (elapsedSeconds > 35) return "Local context is ready; waiting for the final answer...";
  return "Retrieving evidence and drafting...";
}

function friendlyErrorMessage(err) {
  const message = String(err?.message || err || "");
  if (/failed to fetch|networkerror|load failed/i.test(message)) {
    return "The browser lost the connection to the local server. Check that it is still running on port 8787 and press Regenerate.";
  }
  if (/timeout/i.test(message)) {
    return "The local server is taking too long to respond. The query may still be running; press Regenerate if nothing appears in a few seconds.";
  }
  return message || "Network error with no details";
}

async function askWithJob(payload, pending, chat, signal) {
  const startedAt = Date.now();
  const start = await fetchJson("/ask/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
    timeout: ASK_START_TIMEOUT_MS,
  });
  if (!start.job_id) throw new Error("The server did not return a job id");
  state.activeJobId = start.job_id;

  let misses = 0;
  while (true) {
    await sleep(ASK_POLL_INTERVAL_MS, signal);
    const elapsedSeconds = Math.round((Date.now() - startedAt) / 1000);
    try {
      const job = await fetchJson(`/ask/jobs/${encodeURIComponent(start.job_id)}`, {
        signal,
        timeout: ASK_POLL_TIMEOUT_MS,
        retries: 1,
        retryDelay: 400,
      });
      misses = 0;
      if (job.status === "done") return job.result || {};
      if (job.status === "error") throw new Error(job.error || "The query failed on the server");
      if (job.status === "cancelled") throw abortError();
      pending.content = progressText(job.status, elapsedSeconds);
      chat.updatedAt = nowIso();
      saveState();
      renderAll();
    } catch (err) {
      if (err.name === "AbortError" || signal?.aborted) throw abortError();
      misses += 1;
      if (misses >= ASK_POLL_MAX_MISSES) {
        throw new Error("Could not reconnect to the local server. Check that the web app is still running on port 8787.");
      }
      pending.content = progressText("running", elapsedSeconds, misses);
      chat.updatedAt = nowIso();
      saveState();
      renderAll();
    }
  }
}

async function submitQuestion(rawQuestion, options = {}) {
  if (state.busy) return;
  const q = String(rawQuestion ?? "").trim();
  if (!q) return;

  let chat = activeChat() || createChat();
  if (!options.regenerate) {
    chat.messages.push({ role: "user", content: q, createdAt: nowIso() });
    if (chat.messages.filter((m) => m.role === "user").length === 1) {
      chat.title = titleFromQuestion(q);
    }
  }

  const pending = {
    role: "assistant",
    content: state.mode === "context" ? "Retrieving local context..." : "Retrieving evidence and drafting...",
    pending: true,
    createdAt: nowIso(),
  };
  chat.messages.push(pending);
  chat.updatedAt = nowIso();
  chat.refs = [];
  chat.engine = currentEngineLabel();
  state.scrollIntent = "bottom";
  saveState();
  renderAll();
  setBusy(true);

  state.aborter = new AbortController();
  const history = visibleHistory(chat);
  try {
    const payload = {
      question: q,
      history,
      mode: state.mode,
      top_k: Number(els.topK.value || 24),
      model: currentModel(),
      use_codex: state.mode !== "context" && currentModel() !== "local_rag",
      language: "auto",
    };
    const data = await askWithJob(payload, pending, chat, state.aborter.signal);

    pending.pending = false;
    if (state.mode === "context") {
      pending.content = JSON.stringify(data, null, 2);
      chat.refs = data.evidence || [];
      chat.engine = "Local context";
    } else {
      pending.content = data.answer || "No answer was received.";
      chat.refs = data.citations || [];
      chat.engine = data.generated_by === "structured" ? "Structured answer" : currentEngineLabel();
    }
    chat.updatedAt = nowIso();
    state.scrollIntent = "assistant-start";
  } catch (err) {
    pending.pending = false;
    const detail = friendlyErrorMessage(err);
    pending.content = err.name === "AbortError"
      ? "Answer cancelled."
      : `Could not fetch the answer from the local server.\n\nDetail: ${detail}\n\nYou can press Regenerate; the page no longer depends on a single long connection, so brief connection drops should not break the chat.`;
    chat.engine = err.name === "AbortError" ? "Cancelled" : "Error";
    state.scrollIntent = "assistant-start";
  } finally {
    state.activeJobId = null;
    state.aborter = null;
    saveState();
    renderAll();
    setBusy(false);
    focusComposer();
  }
}

function focusComposer() {
  els.question.focus();
  autoSize();
}

function autoSize() {
  els.question.style.height = "0px";
  els.question.style.height = `${Math.min(220, Math.max(48, els.question.scrollHeight))}px`;
}

function updateJumpButton() {
  const distance = els.messages.scrollHeight - els.messages.clientHeight - els.messages.scrollTop;
  els.jumpBottomBtn.classList.toggle("hidden", distance < 120);
}

function scrollMessagesBy(delta) {
  if (els.messages.scrollHeight <= els.messages.clientHeight) return false;
  els.messages.scrollTop += delta;
  updateJumpButton();
  return true;
}

function deleteActiveChat() {
  if (state.chats.length <= 1) {
    const chat = activeChat();
    if (chat) {
      chat.messages = [];
      chat.refs = [];
      chat.title = "New chat";
      chat.engine = "Codex High";
    }
  } else {
    state.chats = state.chats.filter((chat) => chat.id !== state.activeId);
    state.activeId = state.chats[0]?.id || null;
  }
  saveState();
  renderAll();
}

async function copyText(text) {
  const value = String(text ?? "").trim();
  if (!value) {
    toast("Nothing to copy");
    return;
  }
  try {
    if (!navigator.clipboard || !window.isSecureContext) throw new Error("clipboard fallback");
    await navigator.clipboard.writeText(value);
    toast("Copied");
    return;
  } catch {
    const area = document.createElement("textarea");
    area.value = value;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.left = "-9999px";
    area.style.top = "0";
    document.body.appendChild(area);
    area.focus();
    area.select();
    try {
      const ok = document.execCommand("copy");
      toast(ok ? "Copied" : "Could not copy");
    } catch {
      toast("Could not copy");
    } finally {
      document.body.removeChild(area);
    }
  }
}

function exportChat() {
  const chat = activeChat();
  if (!chat) return;
  const lines = [`# ${chat.title}`, ""];
  for (const msg of chat.messages.filter((m) => !m.pending)) {
    lines.push(`## ${msg.role === "user" ? "User" : "TIPS GPT"}`);
    lines.push(msg.content);
    lines.push("");
  }
  if (chat.refs?.length) {
    lines.push("## References");
    for (const ref of chat.refs) {
      lines.push(`- [${ref.n ?? ref.ref}] ${ref.label ?? ref.citation ?? ref.title ?? "Reference"} ${ref.local_path ?? ""}`);
    }
  }
  const blob = new Blob([lines.join("\n")], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${chat.title.replace(/[^a-z0-9]+/gi, "-").slice(0, 50) || "tips-chat"}.md`;
  link.click();
  URL.revokeObjectURL(url);
}

function toggleInspector(force) {
  const next = force ?? els.app.dataset.inspector !== "open";
  els.app.dataset.inspector = next ? "open" : "closed";
}

function toggleSidebar() {
  els.app.dataset.sidebar = els.app.dataset.sidebar === "open" ? "closed" : "open";
}

function applyResponsiveLayout() {
  const isSmall = window.matchMedia("(max-width: 820px)").matches;
  const isMedium = window.matchMedia("(max-width: 1180px)").matches;
  if (isSmall) {
    els.app.dataset.sidebar = "closed";
    els.app.dataset.inspector = "closed";
  } else if (isMedium) {
    els.app.dataset.sidebar = "open";
    els.app.dataset.inspector = "closed";
  } else {
    els.app.dataset.sidebar = "open";
    els.app.dataset.inspector = "open";
  }
}

els.askForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const q = els.question.value;
  els.question.value = "";
  autoSize();
  submitQuestion(q);
});

els.question.addEventListener("input", autoSize);
els.messages.addEventListener("scroll", updateJumpButton);
els.jumpBottomBtn.addEventListener("click", () => {
  els.messages.scrollTo({ top: els.messages.scrollHeight, behavior: "smooth" });
});

document.querySelector(".chat-main").addEventListener("wheel", (event) => {
  const interactive = event.target.closest("textarea, input, select, button, a");
  if (interactive) return;
  if (scrollMessagesBy(event.deltaY)) event.preventDefault();
}, { passive: false });

els.question.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    els.askForm.requestSubmit();
  }
});

document.querySelectorAll(".mode").forEach((btn) => {
  btn.addEventListener("click", () => setMode(btn.dataset.mode));
});

[els.modelPreset, els.modelPresetMobile].filter(Boolean).forEach((select) => {
  select.addEventListener("change", () => setModelPreset(select.value));
});

els.stopBtn.addEventListener("click", () => {
  const jobId = state.activeJobId;
  state.aborter?.abort();
  if (jobId) {
    fetch(`/ask/jobs/${encodeURIComponent(jobId)}`, { method: "DELETE" }).catch(() => {});
  }
});
els.newChatBtn.addEventListener("click", () => {
  createChat();
  focusComposer();
});
els.clearChatBtn.addEventListener("click", deleteActiveChat);
els.sidebarBtn.addEventListener("click", toggleSidebar);
els.refsBtn.addEventListener("click", () => toggleInspector());
els.closeRefsBtn.addEventListener("click", () => toggleInspector(false));
els.exportBtn.addEventListener("click", exportChat);
els.chatSearch.addEventListener("input", renderChats);

els.chatList.addEventListener("click", (event) => {
  const item = event.target.closest("[data-chat-id]");
  if (!item) return;
  state.activeId = item.dataset.chatId;
  saveState();
  renderAll();
});

els.quickPrompts.addEventListener("click", (event) => {
  const btn = event.target.closest("[data-prompt]");
  if (!btn) return;
  els.question.value = btn.dataset.prompt;
  autoSize();
  focusComposer();
});

els.messages.addEventListener("click", (event) => {
  const citation = event.target.closest("[data-citation]");
  if (citation) {
    toggleInspector(true);
    const ref = document.getElementById(`ref-${citation.dataset.citation}`);
    if (ref) {
      ref.scrollIntoView({ block: "center", behavior: "smooth" });
      ref.classList.add("pulse");
      window.setTimeout(() => ref.classList.remove("pulse"), 1200);
    }
    return;
  }

  const copy = event.target.closest("[data-copy]");
  if (copy) {
    const chat = activeChat();
    const msg = chat?.messages[Number(copy.dataset.copy)];
    if (msg) copyText(msg.content);
    return;
  }

  const regen = event.target.closest("[data-regenerate]");
  if (regen) {
    const chat = activeChat();
    const index = Number(regen.dataset.regenerate);
    const previousUser = [...chat.messages.slice(0, index)].reverse().find((msg) => msg.role === "user");
    if (!previousUser) return;
    chat.messages = chat.messages.slice(0, index);
    submitQuestion(previousUser.content, { regenerate: true });
  }
});

els.refs.addEventListener("click", (event) => {
  const pathBtn = event.target.closest("[data-copy-path]");
  if (pathBtn) copyText(pathBtn.dataset.copyPath);
});

window.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    els.chatSearch.focus();
  }
  if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key.toLowerCase() === "o") {
    event.preventDefault();
    createChat();
    focusComposer();
  }
});

loadState();
applyResponsiveLayout();
setModelPreset("codex_high");
setMode("answer");
renderAll();
loadManifest();
autoSize();
