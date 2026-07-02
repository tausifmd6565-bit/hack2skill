import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Sidebar from "@/components/Sidebar";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "SignalCortex — Candidate Intelligence",
  description: "Rank candidates by real evidence, not keyword overlap.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={`${inter.className} bg-gray-50 text-slate-900`}>
        <Sidebar />
        <div className="ml-56 min-h-screen flex flex-col">
          {children}
        </div>
      </body>
    </html>
  );
}
