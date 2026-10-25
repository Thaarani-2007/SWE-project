import React, { useState, useEffect, useRef } from "react";
import { marked } from "marked";
import {
  UploadCloud,
  FileCode,
  FolderGit2,
  Send,
  GitBranch,
  CheckCircle2,
  Loader2,
  X,
  Cpu,
  RefreshCw,
  Trash2,
} from "lucide-react";
import "./App.css";

// Configure marked for GitHub-flavored markdown with line breaks
marked.setOptions({
  gfm: true,
  breaks: true,
});

const API_BASE = "http://localhost:8000";

interface Repository {
  id: string;
  name: string;
  status: string;
  created_at?: string;
}

interface IngestionStatus {
  repo_id: string;
  status: string;
  stage: number;
  percentage: number;
  message: string;
  symbol_count?: number;
  file_count?: number;
}

interface Message {
  id: string;
  sender: "user" | "assistant";
  text: string;
  sources?: any[];
  graphRelations?: {
    callers: string[];
    callees: string[];
    data_flows: any[];
  };
  citations?: any[];
  isGrounded?: boolean;
}

export default function App() {
  const [repos, setRepos] = useState<Repository[]>([]);
  const [selectedRepoId, setSelectedRepoId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"upload" | "chat">("upload");

  // Upload & Ingestion State
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [ingestionStatus, setIngestionStatus] = useState<IngestionStatus | null>(null);

  // Chat State
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState("");
  const [isQuerying, setIsQuerying] = useState(false);

  // Modals & Drawers
  const [activeCitation, setActiveCitation] = useState<{ file: string; start: number; end: number } | null>(null);
  const [fileContent, setFileContent] = useState<string | null>(null);
  const [fileContentLoading, setFileContentLoading] = useState(false);
  const [activeGraphInspect, setActiveGraphInspect] = useState<{
    sources: any[];
    relations: { callers: string[]; callees: string[]; data_flows: any[] };
  } | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);

  // Fetch repositories on mount
  useEffect(() => {
    fetchRepositories();
  }, []);

  const fetchRepositories = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/repos`);
      if (res.ok) {
        const data = await res.json();
        setRepos(data);
        if (data.length > 0 && !selectedRepoId) {
          setSelectedRepoId(data[0].id);
          setActiveTab("chat");
        }
      }
    } catch (err) {
      console.warn("Backend not yet connected:", err);
    }
  };

  // Handle Repository Deletion
  const handleDeleteRepo = async (e: React.MouseEvent, repoId: string, repoName: string) => {
    e.stopPropagation();
    if (!window.confirm(`Are you sure you want to delete repository "${repoName}"? This will permanently remove its files, code graph, and vector indexes.`)) {
      return;
    }

    try {
      const res = await fetch(`${API_BASE}/api/repos/${repoId}`, {
        method: "DELETE",
      });
      if (res.ok) {
        if (selectedRepoId === repoId) {
          setSelectedRepoId(null);
          setMessages([]);
        }
        await fetchRepositories();
      } else {
        alert("Failed to delete repository from server.");
      }
    } catch (err) {
      console.error("Delete error:", err);
      alert("Error contacting backend server.");
    }
  };

  // Scroll to bottom on new message
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isQuerying]);

  // Handle File Upload
  const handleFileUpload = async (file: File) => {
    if (!file.name.endsWith(".zip")) {
      alert("Please upload a .zip archive of a Python codebase.");
      return;
    }

    setUploading(true);
    setIngestionStatus({
      repo_id: "",
      status: "EXTRACTING",
      stage: 1,
      percentage: 10,
      message: "Uploading archive to server...",
    });

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/api/upload`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error(`Upload failed with status ${res.status}`);
      }

      const data = await res.json();
      const repoId = data.repo_id;
      pollIngestionStatus(repoId);
    } catch (err: any) {
      setUploading(false);
      setIngestionStatus({
        repo_id: "",
        status: "FAILED",
        stage: 0,
        percentage: 0,
        message: err.message || "Failed to upload archive.",
      });
    }
  };

  // Poll Ingestion Status
  const pollIngestionStatus = (repoId: string) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/status/${repoId}`);
        if (res.ok) {
          const statusData: IngestionStatus = await res.json();
          setIngestionStatus(statusData);

          if (statusData.status === "READY") {
            clearInterval(interval);
            setUploading(false);
            fetchRepositories();
            setSelectedRepoId(repoId);
            setTimeout(() => {
              setActiveTab("chat");
            }, 800);
          } else if (statusData.status === "FAILED") {
            clearInterval(interval);
            setUploading(false);
          }
        }
      } catch (err) {
        console.warn("Status poll error:", err);
      }
    }, 1500);
  };

  // Handle Query Submission with Streaming SSE
  const handleSendMessage = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!inputValue.trim() || !selectedRepoId || isQuerying) return;

    const question = inputValue.trim();
    setInputValue("");

    const userMessage: Message = {
      id: Date.now().toString(),
      sender: "user",
      text: question,
    };

    const assistantMsgId = (Date.now() + 1).toString();
    const assistantMessage: Message = {
      id: assistantMsgId,
      sender: "assistant",
      text: "",
      sources: [],
      graphRelations: { callers: [], callees: [], data_flows: [] },
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setIsQuerying(true);

    try {
      const response = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repo_id: selectedRepoId, question }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`Query failed: ${response.statusText}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const block of lines) {
          if (!block.trim()) continue;
          const eventMatch = block.match(/event:\s*([a-zA-Z0-9_]+)/);
          const dataMatch = block.match(/data:\s*(.+)/s);

          if (eventMatch && dataMatch) {
            const eventType = eventMatch[1];
            try {
              const data = JSON.parse(dataMatch[1]);

              if (eventType === "sources") {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMsgId
                      ? { ...m, sources: data.sources, graphRelations: data.graph_relations }
                      : m
                  )
                );
              } else if (eventType === "token") {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMsgId ? { ...m, text: m.text + data.token } : m
                  )
                );
              } else if (eventType === "done") {
                setMessages((prev) =>
                  prev.map((m) =>
                    m.id === assistantMsgId
                      ? {
                          ...m,
                          citations: data.citations,
                          isGrounded: data.is_grounded,
                          graphRelations: data.relations || m.graphRelations,
                        }
                      : m
                  )
                );
              }
            } catch (pErr) {
              console.warn("Error parsing SSE block:", pErr);
            }
          }
        }
      }
    } catch (err: any) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsgId
            ? { ...m, text: `[Error generating answer: ${err.message}]` }
            : m
        )
      );
    } finally {
      setIsQuerying(false);
    }
  };

  // Load Source File Content for Citation Modal
  const openCitationModal = async (file: string, start: number, end: number) => {
    if (!selectedRepoId) return;
    setActiveCitation({ file, start, end });
    setFileContentLoading(true);
    setFileContent(null);

    try {
      const res = await fetch(`${API_BASE}/api/file_content?repo_id=${selectedRepoId}&path=${encodeURIComponent(file)}`);
      if (res.ok) {
        const data = await res.json();
        setFileContent(data.content);
      } else {
        setFileContent("# File content could not be retrieved from repository.");
      }
    } catch (e: any) {
      setFileContent(`# Error loading file: ${e.message}`);
    } finally {
      setFileContentLoading(false);
    }
  };

  // Render Markdown with clickable citation pills: [file.py:10-20]
  const renderMessageContent = (text: string, isUser: boolean = false) => {
    if (!text) return null;

    if (isUser) {
      return <div className="user-message-content">{text}</div>;
    }

    // 1. Separate smashed lists if the LLM output points on the same line
    let processed = text
      .replace(/([^\n])\s+(\d+\.\s+\*\*)/g, "$1\n\n$2")
      .replace(/([^\n])\s+(-\s+\*\*)/g, "$1\n\n- $2")
      .replace(/([^\n])\s+(-\s+`)/g, "$1\n\n- $2");

    // 2. Transform citations like [file.py:10-20] into interactive citation pill HTML
    const citationRegex = /\[([a-zA-Z0-9_\-\./\\]+\.py):(\d+)(?:-(\d+))?\]/g;
    processed = processed.replace(citationRegex, (_, file, startStr, endStr) => {
      const start = parseInt(startStr, 10);
      const end = endStr ? parseInt(endStr, 10) : start;
      const label = end !== start ? `${file}:${start}-${end}` : `${file}:${start}`;
      return `<button type="button" class="citation-pill" data-file="${file}" data-start="${start}" data-end="${end}" title="View ${file} (lines ${start}-${end})"><span class="pill-tag">FILE</span><span>${label}</span></button>`;
    });

    // 3. Parse Markdown using marked
    const parsedHtml = marked.parse(processed, { breaks: true, gfm: true }) as string;

    return (
      <div
        className="markdown-body"
        onClick={(e) => {
          const btn = (e.target as HTMLElement).closest(".citation-pill");
          if (btn) {
            e.preventDefault();
            e.stopPropagation();
            const file = btn.getAttribute("data-file");
            const start = parseInt(btn.getAttribute("data-start") || "1", 10);
            const end = parseInt(btn.getAttribute("data-end") || "1", 10);
            if (file) {
              openCitationModal(file, start, end);
            }
          }
        }}
        dangerouslySetInnerHTML={{ __html: parsedHtml }}
      />
    );
  };

  const selectedRepo = repos.find((r) => r.id === selectedRepoId);

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo-icon">CR</div>
          <div>
            <div className="sidebar-title">Code RAG</div>
            <div className="sidebar-sub">AST & Graph Assisted RAG</div>
          </div>
        </div>

        <div className="sidebar-actions">
          <button
            className="btn-upload-new"
            onClick={() => {
              setActiveTab("upload");
              setIngestionStatus(null);
            }}
          >
            <UploadCloud size={16} />
            Upload Codebase
          </button>
        </div>

        <div className="sidebar-section-title">Indexed Repositories ({repos.length})</div>
        <div className="repo-list">
          {repos.map((repo) => (
            <div
              key={repo.id}
              className={`repo-item ${selectedRepoId === repo.id && activeTab === "chat" ? "active" : ""}`}
              onClick={() => {
                setSelectedRepoId(repo.id);
                setActiveTab("chat");
              }}
            >
              <div className="repo-item-title">
                <span className="repo-name-text">{repo.name}</span>
                <div className="repo-badge-actions">
                  <span className={`repo-status-badge ${repo.status === "READY" ? "badge-ready" : "badge-processing"}`}>
                    {repo.status}
                  </span>
                  <button
                    type="button"
                    className="btn-delete-repo"
                    title={`Delete repository ${repo.name}`}
                    onClick={(e) => handleDeleteRepo(e, repo.id, repo.name)}
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              </div>
              <div className="repo-item-meta">
                <span>ID: {repo.id}</span>
                {repo.created_at && <span>{new Date(repo.created_at).toLocaleDateString()}</span>}
              </div>
            </div>
          ))}

          {repos.length === 0 && (
            <div style={{ padding: "1.5rem 0.5rem", color: "var(--text-dim)", fontSize: "0.8rem", textAlign: "center" }}>
              No repositories indexed yet. Upload a .zip codebase to get started.
            </div>
          )}
        </div>
      </aside>

      {/* Main Area */}
      <main className="main-content">
        <header className="main-header">
          <div className="active-repo-badge">
            <FolderGit2 size={16} />
            <span>Active Repository:</span>
            <span className="active-repo-name">{selectedRepo ? selectedRepo.name : "None Selected"}</span>
          </div>
          <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
            <button
              onClick={fetchRepositories}
              className="btn-graph-inspect"
              title="Refresh repositories"
            >
              <RefreshCw size={14} />
              Refresh
            </button>
          </div>
        </header>

        {/* Screen 1: Uploader */}
        {activeTab === "upload" && (
          <div className="uploader-screen">
            <div className="uploader-card">
              <h2 className="uploader-title">Upload Codebase Archive</h2>
              <p className="uploader-desc">
                Upload a ZIP file containing your Python codebase. The system extracts symbols, resolves call graphs, traces data-flow, and indexes embeddings into ChromaDB.
              </p>

              <div
                className={`dropzone ${isDragging ? "drag-active" : ""}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  setIsDragging(false);
                  if (e.dataTransfer.files?.[0]) {
                    handleFileUpload(e.dataTransfer.files[0]);
                  }
                }}
                onClick={() => fileInputRef.current?.click()}
              >
                <div className="dropzone-icon">
                  <UploadCloud size={28} />
                </div>
                <div style={{ textAlign: "center" }}>
                  <div style={{ fontWeight: 600, fontSize: "0.95rem" }}>
                    Click to browse or drop ZIP file here
                  </div>
                  <div style={{ color: "var(--text-dim)", fontSize: "0.8rem", marginTop: "0.25rem" }}>
                    Supports .zip archives containing Python (.py) source files (up to 150MB)
                  </div>
                </div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".zip"
                  style={{ display: "none" }}
                  onChange={(e) => {
                    if (e.target.files?.[0]) {
                      handleFileUpload(e.target.files[0]);
                    }
                  }}
                />
              </div>

              {/* Multi-stage Progress Tracker */}
              {ingestionStatus && (
                <div className="progress-container">
                  <div className="progress-bar-bg">
                    <div className="progress-bar-fill" style={{ width: `${ingestionStatus.percentage}%` }} />
                  </div>

                  <div className="stages-stepper">
                    {[
                      { num: 1, label: "Extracting" },
                      { num: 2, label: "AST & Flow" },
                      { num: 3, label: "Code Graph" },
                      { num: 4, label: "Chroma Index" },
                    ].map((step) => {
                      const isComplete = ingestionStatus.stage > step.num || ingestionStatus.status === "READY";
                      const isActive = ingestionStatus.stage === step.num && ingestionStatus.status !== "READY";
                      return (
                        <div
                          key={step.num}
                          className={`stage-step ${isComplete ? "completed" : ""} ${isActive ? "active" : ""}`}
                        >
                          <div className="stage-dot">
                            {isComplete ? <CheckCircle2 size={14} /> : step.num}
                          </div>
                          <span className="stage-label">[{step.num}] {step.label}</span>
                        </div>
                      );
                    })}
                  </div>

                  <div className="progress-message">
                    {uploading && <Loader2 size={14} style={{ display: "inline", marginRight: "6px", animation: "spin 1s linear infinite" }} />}
                    {ingestionStatus.message}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Screen 2: Chat Workspace */}
        {activeTab === "chat" && (
          <div className="chat-screen">
            <div className="chat-thread">
              {messages.length === 0 && (
                <div className="chat-empty-state">
                  <Cpu size={48} style={{ color: "var(--primary)", opacity: 0.8 }} />
                  <div style={{ fontSize: "1.1rem", fontWeight: 600 }}>Ask anything about {selectedRepo?.name}</div>
                  <div style={{ fontSize: "0.85rem", maxWidth: "450px", textAlign: "center" }}>
                    The system combines semantic vector search with NetworkX call & data-flow graph expansion, reranking source snippets and generating grounded answers with verified citations.
                  </div>
                </div>
              )}

              {messages.map((msg) => (
                <div key={msg.id} className={`message-wrapper ${msg.sender}`}>
                  <div className={`message-avatar ${msg.sender}`}>
                    {msg.sender === "assistant" ? "AI" : "YOU"}
                  </div>

                  <div className="message-bubble">
                    {renderMessageContent(msg.text, msg.sender === "user")}

                    {/* Graph Inspector pill bar */}
                    {msg.sender === "assistant" && msg.sources && msg.sources.length > 0 && (
                      <div className="graph-inspector-bar">
                        <button
                          className="btn-graph-inspect"
                          onClick={() =>
                            setActiveGraphInspect({
                              sources: msg.sources || [],
                              relations: msg.graphRelations || { callers: [], callees: [], data_flows: [] },
                            })
                          }
                        >
                          <GitBranch size={13} />
                          Inspect Graph Context ({msg.sources.length} snippets,{" "}
                          {(msg.graphRelations?.callers?.length || 0)} callers,{" "}
                          {(msg.graphRelations?.callees?.length || 0)} callees)
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {isQuerying && messages[messages.length - 1]?.sender === "user" && (
                <div className="message-wrapper assistant">
                  <div className="message-avatar assistant">AI</div>
                  <div className="message-bubble" style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                    <Loader2 size={16} style={{ animation: "spin 1s linear infinite" }} />
                    <span style={{ color: "var(--text-muted)", fontSize: "0.85rem" }}>
                      Searching vector store & expanding code graph...
                    </span>
                  </div>
                </div>
              )}

              <div ref={chatEndRef} />
            </div>

            {/* Input form */}
            <div className="chat-input-container">
              <form className="chat-input-form" onSubmit={handleSendMessage}>
                <input
                  type="text"
                  className="chat-input"
                  placeholder={
                    selectedRepoId
                      ? `Ask a question about ${selectedRepo?.name}... (e.g. How does user authentication work?)`
                      : "Please upload or select a repository first..."
                  }
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  disabled={!selectedRepoId || isQuerying}
                />
                <button
                  type="submit"
                  className="btn-send"
                  disabled={!inputValue.trim() || !selectedRepoId || isQuerying}
                >
                  <Send size={16} />
                </button>
              </form>
            </div>
          </div>
        )}

        {/* Modal 1: Citation Code Viewer */}
        {activeCitation && (
          <div className="modal-overlay" onClick={() => setActiveCitation(null)}>
            <div className="code-modal" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <div className="modal-title">
                  <FileCode size={16} style={{ color: "var(--accent-cyan)" }} />
                  <span>{activeCitation.file}</span>
                  <span style={{ color: "var(--text-dim)", fontSize: "0.8rem" }}>
                    (Lines {activeCitation.start}-{activeCitation.end})
                  </span>
                </div>
                <button className="btn-close" onClick={() => setActiveCitation(null)}>
                  <X size={18} />
                </button>
              </div>

              <div className="code-viewer-body">
                {fileContentLoading && (
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "center", padding: "2rem" }}>
                    <Loader2 size={24} style={{ animation: "spin 1s linear infinite", color: "var(--primary)" }} />
                  </div>
                )}
                {fileContent && (
                  <div>
                    {fileContent.split("\n").map((line, idx) => {
                      const lineNum = idx + 1;
                      const isHighlighted = lineNum >= activeCitation.start && lineNum <= activeCitation.end;
                      return (
                        <div key={lineNum} className={`code-line ${isHighlighted ? "highlighted" : ""}`}>
                          <span className="line-number">{lineNum}</span>
                          <span className="line-code">{line}</span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Modal 2: Graph Context Inspector */}
        {activeGraphInspect && (
          <div className="modal-overlay" onClick={() => setActiveGraphInspect(null)}>
            <div className="code-modal" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <div className="modal-title">
                  <GitBranch size={16} style={{ color: "var(--accent-cyan)" }} />
                  <span>Graph Inspector & Source Context</span>
                </div>
                <button className="btn-close" onClick={() => setActiveGraphInspect(null)}>
                  <X size={18} />
                </button>
              </div>

              <div className="graph-drawer">
                {/* Reranked Sources */}
                <div>
                  <div className="graph-section-title">Reranked Evidence Snippets ({activeGraphInspect.sources.length})</div>
                  <div className="graph-cards-grid">
                    {activeGraphInspect.sources.map((s, idx) => (
                      <div key={idx} className="graph-node-card">
                        <div className="graph-node-title">{s.name || s.id}</div>
                        <div className="graph-node-subtitle">
                          {s.file} (lines {s.lines})
                        </div>
                        <div style={{ display: "flex", gap: "0.4rem", marginTop: "0.25rem" }}>
                          <span className="repo-status-badge badge-ready">{s.relationship}</span>
                          {s.score > 0 && (
                            <span className="repo-status-badge badge-processing">
                              Score: {s.score.toFixed(3)}
                            </span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Callers */}
                {activeGraphInspect.relations.callers?.length > 0 && (
                  <div>
                    <div className="graph-section-title" style={{ color: "var(--accent-purple)" }}>
                      1-Hop Callers ({activeGraphInspect.relations.callers.length})
                    </div>
                    <div className="graph-cards-grid">
                      {activeGraphInspect.relations.callers.map((c, idx) => (
                        <div key={idx} className="graph-node-card caller">
                          <div className="graph-node-title">{c}</div>
                          <div className="graph-node-subtitle">Direct caller in code graph</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Callees */}
                {activeGraphInspect.relations.callees?.length > 0 && (
                  <div>
                    <div className="graph-section-title" style={{ color: "var(--accent-cyan)" }}>
                      1-Hop Callees ({activeGraphInspect.relations.callees.length})
                    </div>
                    <div className="graph-cards-grid">
                      {activeGraphInspect.relations.callees.map((c, idx) => (
                        <div key={idx} className="graph-node-card callee">
                          <div className="graph-node-title">{c}</div>
                          <div className="graph-node-subtitle">Function invoked by retrieved symbols</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Data Flows */}
                {activeGraphInspect.relations.data_flows?.length > 0 && (
                  <div>
                    <div className="graph-section-title" style={{ color: "var(--accent-green)" }}>
                      Data-Flow Links ({activeGraphInspect.relations.data_flows.length})
                    </div>
                    <div className="graph-cards-grid">
                      {activeGraphInspect.relations.data_flows.slice(0, 8).map((df, idx) => (
                        <div key={idx} className="graph-node-card dataflow">
                          <div className="graph-node-title">{df.source} ➔ {df.target}</div>
                          <div className="graph-node-subtitle">Flow Type: {df.flow_type}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
