// client/src/ChatPage.jsx
import React, { useState } from "react";
const API_URL = process.env.REACT_APP_API_URL || "http://localhost:4000";

export default function ChatPage() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: "ai",
      text: "Hello! I'm Keynes. Upload a PDF/PNG and ask a question.",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(false);

  const onPick = (e) => setFiles(Array.from(e.target.files || []));

  async function uploadAndIndex() {
    if (!files.length) return { indexed: 0 };
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    const r = await fetch(`${API_URL}/api/upload`, { method: "POST", body: form });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  }

  async function ask(question) {
    const r = await fetch(`${API_URL}/api/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!r.ok) throw new Error(await r.text());
    return r.json();
  }

  const onSend = async (e) => {
    e.preventDefault();
    if (!input.trim() && files.length === 0) return;

    setLoading(true);
    const userMsg = {
      id: Date.now(),
      sender: "user",
      text: input || "(no text)",
      timestamp: new Date(),
    };
    setMessages((m) => [...m, userMsg]);

    try {
      if (files.length) {
        await uploadAndIndex();
        setFiles([]);
      }
      const { answer, sources } = await ask(
        input || "Analyze the uploaded documents/images."
      );
      const aiMsg = {
        id: Date.now() + 1,
        sender: "ai",
        text: answer || "No response.",
        timestamp: new Date(),
      };
      setMessages((m) => [...m, aiMsg]);
      if (sources) console.log("sources:", sources);
    } catch (err) {
      console.error(err);
      setMessages((m) => [
        ...m,
        {
          id: Date.now() + 2,
          sender: "ai",
          text: "Sorry, that failed.",
          timestamp: new Date(),
        },
      ]);
    } finally {
      setLoading(false);
      setInput("");
    }
  };

  return (
    <div className="chat-container">
      <div className="uploaded-files" style={{ marginBottom: 12 }}>
        <input
          id="file"
          type="file"
          multiple
          accept=".pdf,.png,.jpg,.jpeg"
          onChange={onPick}
        />
        {files.length ? <small>{files.length} file(s) selected</small> : null}
      </div>

      <div className="chat-messages">
        {messages.map((m) => (
          <div key={m.id} className={`message ${m.sender}`}>
            <div className="bubble">
              <div>{m.text}</div>
              <small>
                {m.timestamp.toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </small>
            </div>
          </div>
        ))}
      </div>

      <form onSubmit={onSend} style={{ marginTop: 12 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about the files…"
          disabled={loading}
        />
        <button disabled={loading || (!input.trim() && files.length === 0)}>
          {loading ? "…" : "Send"}
        </button>
      </form>
    </div>
  );
}
