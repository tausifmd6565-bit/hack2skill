import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",       // export as static HTML/JS/CSS
  distDir: "out",         // output to frontend/out/
  trailingSlash: true,    // so /ranking/ works as a file path
  images: { unoptimized: true },
};

export default nextConfig;
