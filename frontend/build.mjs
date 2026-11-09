import { build } from 'esbuild';
import fs from 'node:fs';
const out = '../app/static';
fs.mkdirSync(`${out}/assets`, { recursive: true });
await build({entryPoints:['src/main.jsx'],bundle:true,minify:true,outfile:`${out}/assets/app.js`,loader:{'.jsx':'jsx'},define:{'process.env.NODE_ENV':'"production"'},legalComments:'linked'});
fs.copyFileSync('src/styles.css', `${out}/assets/app.css`);
fs.copyFileSync('../THIRD_PARTY_LICENSES.txt', `${out}/assets/THIRD_PARTY_LICENSES.txt`);
fs.writeFileSync(`${out}/index.html`, `<!doctype html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="referrer" content="no-referrer"><title>Clinical Workspace</title><link rel="stylesheet" href="/assets/app.css"><script defer src="/assets/app.js"></script></head><body><div id="root"></div></body></html>`);
