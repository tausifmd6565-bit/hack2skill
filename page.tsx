"use client";
import { useState, useRef } from "react";
import Topbar from "@/components/Topbar";
import { analyzeJD } from "@/lib/api";
import { useRouter } from "next/navigation";
import { Upload, CheckCircle, AlertCircle, Loader2 } from "lucide-react";

const DEFAULT_JD = `We are looking for a Senior AI Engineer to join our founding team at Redrob AI (Series A AI-native talent platform).

You will own the candidate ranking and retrieval infrastructure end-to-end.

Requirements:
- 5-9 years of industry experience in ML/AI engineering
- Production experience with embeddings, vector databases, and hybrid search
- Strong Python engineering skills (FastAPI, Docker, CI/CD)
- Experience with ranking evaluation metrics: NDCG, MRR, MAP
- Built or maintained search, recommendation, or ranking systems in production
- Product company or startup background

Nice to have:
- LLM fine-tuning or instruction tuning experience
- Learning-to-rank models (LambdaMART, XGBoost rank)
- HR-tech or recruiting-tech domain experience

Work mode: Hybrid — Bangalore/Pune preferred`;

const STEPS = ["Upload JD", "Analyze Role DNA", "Rank Candidates", "Export Shortlist"];

export default function UploadPage() {
  const router = useRouter();
  const [jdText, setJdText] = useState(DEFAULT_JD);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [step, setStep] = useState(0);
  const fileRef = useRef<HTMLInputElement>(null);

  const handleFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) setFile(e.target.files[0]);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f) setFile(f);
  };

  const handleAnalyze = async () => {
    if (!jdText.trim()) { setError("Please enter a job description."); return; }
    if (!file) { setError("Please upload a candidate CSV file."); return; }
    setError("");
    setLoading(true);
    setStep(1);
    try {
      await analyzeJD(jdText, file);
      setStep(3);
      setTimeout(() => router.push("/ranking"), 600);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Analysis failed. Is the backend running on port 8000?";
      setError(msg);
      setStep(0);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Topbar title="Upload JD & Dataset" />
      <main className="p-6 flex-1">
        <div className="mb-5">
          <h2 className="text-xl font-extrabold text-slate-900 tracking-tight">Upload JD & Dataset</h2>
          <p className="text-sm text-gray-500 mt-0.5">Provide a job description and candidate data to begin intelligence analysis.</p>
        </div>

        {/* Stepper */}
        <div className="flex items-center gap-0 mb-7">
          {STEPS.map((s, i) => (
            <div key={s} className="flex items-center">
              <div className="flex items-center gap-2">
                <div className={`w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold ${
                  i < step ? "bg-green-500 text-white" : i === step ? "bg-blue-600 text-white" : "bg-gray-200 text-gray-400"
                }`}>
                  {i < step ? <CheckCircle size={14} /> : i + 1}
                </div>
                <span className={`text-xs font-semibold ${i === step ? "text-slate-900" : "text-gray-400"}`}>{s}</span>
              </div>
              {i < STEPS.length - 1 && <div className={`h-0.5 w-10 mx-2 ${i < step ? "bg-green-400" : "bg-gray-200"}`} />}
            </div>
          ))}
        </div>

        <div className="grid grid-cols-2 gap-6 mb-6">
          {/* JD text */}
          <div className="bg-white border border-gray-200 rounded-xl p-5">
            <div className="text-sm font-bold text-slate-900 mb-1">Job Description</div>
            <div className="text-xs text-gray-500 mb-3">Paste the full JD. The AI will extract Role DNA automatically.</div>
            <textarea
              value={jdText}
              onChange={(e) => setJdText(e.target.value)}
              rows={14}
              className="w-full border border-gray-200 rounded-lg p-3 text-xs text-slate-700 font-mono resize-none outline-none focus:border-blue-400 transition-colors"
            />
            <div className="flex gap-2 mt-3">
              <button className="text-xs border border-gray-200 rounded-lg px-3 py-1.5 font-semibold text-gray-600 hover:bg-gray-50">
                Upload .txt / .pdf
              </button>
            </div>
          </div>

          {/* Right column */}
          <div className="space-y-4">
            {/* File upload */}
            <div className="bg-white border border-gray-200 rounded-xl p-5">
              <div className="text-sm font-bold text-slate-900 mb-3">Candidate Dataset</div>
              <div
                onDrop={handleDrop}
                onDragOver={(e) => e.preventDefault()}
                onClick={() => fileRef.current?.click()}
                className="border-2 border-dashed border-gray-200 rounded-xl p-8 text-center cursor-pointer hover:border-blue-400 hover:bg-blue-50/30 transition-colors"
              >
                <Upload size={28} className="mx-auto text-gray-300 mb-2" />
                <div className="text-sm font-bold text-slate-700">
                  {file ? file.name : "Drop candidates.csv or candidates.json"}
                </div>
                <div className="text-xs text-gray-400 mt-1">Supports CSV, JSON · Max 50MB</div>
                <button className="mt-3 text-xs font-semibold text-white bg-blue-600 px-4 py-1.5 rounded-lg hover:bg-blue-700">
                  Browse File
                </button>
              </div>
              <input ref={fileRef} type="file" accept=".csv,.json" className="hidden" onChange={handleFile} />
            </div>

            {/* Optional evidence fields */}
            <div className="bg-white border border-gray-200 rounded-xl p-5">
              <div className="text-sm font-bold text-slate-900 mb-3">Optional Evidence Sources</div>
              <div className="grid grid-cols-2 gap-2">
                {[["GitHub URLs", "Field: github_url"], ["Portfolio URLs", "Field: portfolio_url"],
                  ["Platform Activity", "last_login, response_rate"], ["Availability Data", "notice_period, relocation"]
                ].map(([title, sub]) => (
                  <div key={title} className="bg-gray-50 border border-gray-100 rounded-lg p-2.5 text-xs">
                    <div className="font-bold text-slate-800">{title}</div>
                    <div className="text-gray-400 mt-0.5">{sub}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Dataset quality */}
        <div className="bg-white border border-gray-200 rounded-xl p-5 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div className="text-sm font-bold text-slate-900">Dataset Quality Check</div>
            <span className="text-[11px] font-semibold bg-green-100 text-green-700 px-2 py-1 rounded-full">✓ Ready to Analyze</span>
          </div>
          <div className="grid grid-cols-5 gap-3">
            {[["500", "Total profiles"], ["—", "With GitHub"], ["—", "With portfolio"], ["—", "With activity"], ["—", "Missing fields"]].map(([v, l]) => (
              <div key={l} className="bg-gray-50 rounded-lg p-3 text-center">
                <div className="text-xl font-extrabold text-slate-900">{v}</div>
                <div className="text-[11px] text-gray-500 mt-0.5">{l}</div>
              </div>
            ))}
          </div>
          {file && (
            <div className="mt-3 flex items-center gap-2 text-xs text-green-700 bg-green-50 rounded-lg p-2.5">
              <CheckCircle size={14} /> File loaded: <strong>{file.name}</strong>
            </div>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="flex items-center gap-2 text-xs text-red-700 bg-red-50 border border-red-200 rounded-lg p-3 mb-4">
            <AlertCircle size={14} /> {error}
          </div>
        )}

        {/* CTA */}
        <div className="text-center">
          <button
            onClick={handleAnalyze}
            disabled={loading}
            className="inline-flex items-center gap-2 text-sm font-bold text-white bg-blue-600 px-8 py-3 rounded-xl hover:bg-blue-700 disabled:opacity-60 disabled:cursor-not-allowed transition-colors shadow-sm"
          >
            {loading ? <><Loader2 size={16} className="animate-spin" /> Analyzing...</> : "⚡ Run Candidate Intelligence"}
          </button>
          <div className="text-xs text-gray-400 mt-2">Estimated time: ~45 seconds for 500 profiles</div>
        </div>
      </main>
    </>
  );
}
