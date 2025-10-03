// server/index.js
import express from "express";
import cors from "cors";
import multer from "multer";
import OpenAI from "openai";
import dotenv from "dotenv";
import { createRequire } from "module";

const require = createRequire(import.meta.url);
const pdfParseModule = require("pdf-parse");
const pdfParse = pdfParseModule.pdf; // Grab the pdf function from the object

// Optional sanity check
if (typeof pdfParse !== "function") {
  console.error("pdf-parse type:", typeof pdfParse, pdfParse);
  throw new Error("Expected pdf-parse.pdf to be a function");
}


dotenv.config();

const app = express();
app.use(cors());
app.use(express.json({ limit: "25mb" }));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 },
});

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ---- In-memory vector store ----
let DOCS = []; // [{id, text, embedding, meta:{filename,page?,type}}]

function cosSim(a, b) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function chunkText(text, chunkSize = 1500, overlap = 200) {
  const chunks = [];
  let i = 0;
  while (i < text.length) {
    const slice = text.slice(i, i + chunkSize);
    chunks.push(slice);
    i += chunkSize - overlap;
  }
  return chunks;
}

// ⬇️ Extract text from PDF buffer using pdf-parse
async function extractPdfText(buffer) {
  try {
    if (!Buffer.isBuffer(buffer) && !(buffer instanceof Uint8Array)) {
      throw new TypeError("extractPdfText: expected Buffer/Uint8Array");
    }
    const out = await pdfParse(buffer);   // <-- ✅ no .default here
    return out?.text ?? "";
  } catch (error) {
    console.error('PDF parsing error:', error);
    return "Error processing PDF file. Please try a different PDF or upload an image instead.";
  }
}

// Vision: caption image to text we can embed
async function captionImageDataUrl(dataUrl) {
  const resp = await openai.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [
      {
        role: "user",
        content: [
          {
            type: "text",
            text: "Briefly describe this image in 2-4 sentences. If there are numbers or tickers, include them.",
          },
          { type: "image_url", image_url: { url: dataUrl } },
        ],
      },
    ],
  });
  return resp.choices[0]?.message?.content?.trim() || "";
}

async function embedBatch(texts) {
  if (texts.length === 0) return [];
  const { data } = await openai.embeddings.create({
    model: "text-embedding-3-large",
    input: texts,
  });
  return data.map((x) => x.embedding);
}

// ---- Upload & index (PDFs & images) ----
app.post("/api/upload", upload.array("files"), async (req, res) => {
  try {
    if (!req.files?.length) return res.status(400).json({ error: "No files" });

    const toIndex = []; // [{text, meta}]
    for (const f of req.files) {
      if (f.mimetype === "application/pdf") {
        // ⬇️ use pdfjs-dist instead of pdf-parse
        const raw = await extractPdfText(f.buffer);
        const clean = raw.replace(/\s+\n/g, "\n").trim();
        const chunks = chunkText(clean);
        chunks.forEach((c, idx) =>
          toIndex.push({
            text: c,
            meta: { filename: f.originalname, type: "pdf", page: idx + 1 },
          })
        );
      } else if (f.mimetype.startsWith("image/")) {
        const base64 = f.buffer.toString("base64");
        const dataUrl = `data:${f.mimetype};base64,${base64}`;
        const caption = await captionImageDataUrl(dataUrl);
        if (caption)
          toIndex.push({
            text: `${f.originalname} (image caption): ${caption}`,
            meta: { filename: f.originalname, type: "image" },
          });
      }
    }

    const embeddings = await embedBatch(toIndex.map((x) => x.text));
    embeddings.forEach((emb, i) => {
      DOCS.push({
        id: `${Date.now()}_${i}_${Math.random().toString(36).slice(2, 8)}`,
        text: toIndex[i].text,
        embedding: emb,
        meta: toIndex[i].meta,
      });
    });

    res.json({ indexed: toIndex.length, totalIndexed: DOCS.length });
  } catch (err) {
    console.error("Upload/index error:", err);
    res.status(500).json({ error: "Failed to index files" });
  }
});

// ---- Ask: retrieve top-k and answer with context ----
app.post("/api/ask", async (req, res) => {
  try {
    const { question, k = 6 } = req.body;
    if (!question?.trim()) return res.status(400).json({ error: "No question" });

    const qEmb = (
      await openai.embeddings.create({
        model: "text-embedding-3-large",
        input: question,
      })
    ).data[0].embedding;

    const scored = DOCS
      .map((d) => ({ d, score: cosSim(qEmb, d.embedding) }))
      .sort((a, b) => b.score - a.score)
      .slice(0, k);

    const context = scored
      .map(
        ({ d, score }, i) =>
          `#${i + 1} [${score.toFixed(3)}] (${d.meta.type}:${
            d.meta.filename
          }${d.meta.page ? ` p.${d.meta.page}` : ""})\n${d.text}`
      )
      .join("\n\n---\n\n");

    const prompt = `You are Keynes, a value investing assistant. Use ONLY the context to answer; if missing, say so briefly.\n\nContext:\n${context}\n\nQuestion: ${question}\n\nAnswer:`;

    const resp = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: "Be concise, precise, and cite line items if possible." },
        { role: "user", content: prompt },
      ],
      temperature: 0.2,
      max_tokens: 600,
    });

    res.json({
      answer: resp.choices[0]?.message?.content || "No answer.",
      sources: scored.map(({ d, score }) => ({
        meta: d.meta,
        score: Number(score.toFixed(3)),
      })),
    });
  } catch (err) {
    console.error("Ask error:", err);
    res.status(500).json({ error: "Ask failed" });
  }
});

const PORT = process.env.PORT || 4000;
app.listen(PORT, () =>
  console.log(`RAG server listening on http://localhost:${PORT}`)
);
