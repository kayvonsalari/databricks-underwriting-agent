// Same lenient parsing as the notebook's parse_json: take the text between the
// first "{" and the last "}". Returns it as a JSON string so is-json can check it.
// If there is no JSON at all, the raw text is returned and is-json fails.
module.exports = (output) => {
  const text = String(output);
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  return start >= 0 && end > start ? text.slice(start, end + 1) : text;
};
