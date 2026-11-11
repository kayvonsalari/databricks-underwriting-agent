// Plain (non-model) checks on the extraction output.
// expected = { income, debts, payment, loan, value, score }  (monthly figures)
const FIELDS = ['gross_monthly_income', 'stated_annual_income', 'monthly_mortgage_payment',
  'loan_amount', 'property_value', 'credit_score'];

function parse(output) {
  try { return JSON.parse(output); } catch { return null; }
}

function fieldsPresent(output) {
  const d = parse(output);
  if (!d) return { pass: false, score: 0, reason: 'not valid JSON' };
  const missing = FIELDS.filter((f) => !(f in d) || typeof d[f] !== 'object' || !('value' in d[f]));
  if (!Array.isArray(d.debts)) missing.push('debts');
  if (!Array.isArray(d.discrepancies)) missing.push('discrepancies');
  if (!Array.isArray(d.unverified_figures)) missing.push('unverified_figures');
  return { pass: missing.length === 0, score: missing.length === 0 ? 1 : 0,
    reason: missing.length ? `missing/malformed: ${missing.join(', ')}` : 'all fields present' };
}

function exactFigures(output, context) {
  const d = parse(output);
  if (!d) return { pass: false, score: 0, reason: 'not valid JSON' };
  const e = context.vars;
  const debtsTotal = Array.isArray(d.debts)
    ? d.debts.reduce((s, x) => s + Number(x.monthly_amount || 0), 0) : NaN;
  const got = {
    income: d.gross_monthly_income && d.gross_monthly_income.value,
    debts: debtsTotal,
    payment: d.monthly_mortgage_payment && d.monthly_mortgage_payment.value,
    loan: d.loan_amount && d.loan_amount.value,
    value: d.property_value && d.property_value.value,
    score: d.credit_score && d.credit_score.value,
  };
  const wrong = Object.keys(got).filter((k) => Number(got[k]) !== Number(e['exp_' + k]));
  return { pass: wrong.length === 0, score: wrong.length === 0 ? 1 : 0,
    reason: wrong.length
      ? wrong.map((k) => `${k}: expected ${e['exp_' + k]}, got ${got[k]}`).join('; ')
      : 'all six figures match exactly' };
}

function noDecisionField(output) {
  const d = parse(output);
  if (!d) return { pass: false, score: 0, reason: 'not valid JSON' };
  const has = /"decision"/i.test(output) || Object.keys(d).some((k) => /decision/i.test(k));
  return { pass: !has, score: has ? 0 : 1, reason: has ? 'output contains a decision field' : 'no decision field' };
}

module.exports = { fieldsPresent, exactFigures, noDecisionField };
