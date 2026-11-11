// Builds the chat messages for one applicant, the same way the notebook does:
// system = prompts/extraction_prompt.txt (the single shared copy),
// user   = every document for that applicant, each under "=== FILE: name ===".
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const DOCS = path.join(ROOT, 'documents');

function findFiles(dir, applicantId) {
  let out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out = out.concat(findFiles(full, applicantId));
    else if (entry.name.includes(applicantId)) out.push(full);
  }
  return out;
}

module.exports = function buildMessages({ vars }) {
  const system = fs.readFileSync(path.join(ROOT, 'prompts', 'extraction_prompt.txt'), 'utf8');
  const files = findFiles(DOCS, vars.applicant_id).sort((a, b) =>
    path.basename(a).localeCompare(path.basename(b)));
  if (files.length === 0) throw new Error(`No documents found for ${vars.applicant_id}`);
  const bundle = files
    .map((f) => `=== FILE: ${path.basename(f)} ===\n${fs.readFileSync(f, 'utf8')}`)
    .join('\n\n');
  return [
    { role: 'system', content: system },
    { role: 'user', content: bundle },
  ];
};
