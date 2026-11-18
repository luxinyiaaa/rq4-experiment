import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { SpreadsheetFile, Workbook } from '@oai/artifact-tool';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const outDir = `${root}/outputs/rawpage_questions/question_set_v3`;
const readJson = async (name) => JSON.parse(await fs.readFile(`${outDir}/${name}`, 'utf8'));
const readJsonl = async (name) => (await fs.readFile(`${outDir}/${name}`, 'utf8')).split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));

const questionHeaders = ['id','page_id','set','page_type','site','url','type','type_substituted','type_substitution_reason','question','reference_answer','source_location','keypoints','supporting_quotes','false_premise','kb_label_final','needs_kb_check','quote_in_page','in_kb','in_kb_any','kb_other_urls','kb_best_match','similarity_label','live_in_kb','table_based','quote_cut_off','duplicate','duplicate_of','teacher_model'];
const removedHeaders = [...questionHeaders, 'reason_code', 'removal_reason'];
const manualHeaders = ['id','set','kb_label_final','kb_check_note','table_answer_ok','table_note','decision','note'];
const auditHeaders = ['id','set','type','page_type','url','question','reference_answer','answer_fit','single_answer','premise_ok','reason','flagged','manual_decision','manual_note','judge_model'];
const supervisorHeaders = ['id','page_type','site','url','question type','question','reference_answer','keypoints','source quotes','source_location','kb_label_final'];

const main = await readJsonl('questions_main.jsonl');
const extra = await readJsonl('questions_extra.jsonl');
const all = await readJsonl('questions_all.jsonl');
const manual = await readJson('manual_checks.json');
const audit = await readJsonl('audit.jsonl');
const sample = await readJson('review_sample_data.json');
const supData = await readJson('supervisor_data.json');
const sets = Object.fromEntries(['reserve','pending','removed'].map((s) => [s, all.filter((q) => q.set === s)]));

function cell(value) {
  if (value === undefined || value === null) return '';
  if (Array.isArray(value) || (typeof value === 'object' && value !== null)) return JSON.stringify(value);
  if (typeof value === 'boolean') return value ? 'yes' : 'no';
  return value;
}
function matrix(headers, rows) {
  return [headers, ...rows.map((r) => headers.map((h) => cell(r[h])))];
}
function addTableSheet(wb, name, headers, rows, widths = {}) {
  const sheet = wb.worksheets.add(name);
  const values = matrix(headers, rows);
  sheet.getRangeByIndexes(0, 0, values.length, headers.length).values = values;
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  sheet.getRangeByIndexes(0, 0, 1, headers.length).format = {
    fill: '#17365D', font: { bold: true, color: '#FFFFFF' }, wrapText: true, verticalAlignment: 'center'
  };
  sheet.getRangeByIndexes(0, 0, values.length, headers.length).format.wrapText = true;
  sheet.getRangeByIndexes(0, 0, values.length, headers.length).format.verticalAlignment = 'top';
  sheet.getRangeByIndexes(0, 0, 1, headers.length).format.rowHeight = 32;
  for (const [column, width] of Object.entries(widths)) {
    const index = headers.indexOf(column);
    if (index >= 0) sheet.getRangeByIndexes(0, index, Math.max(1, values.length), 1).format.columnWidth = width;
  }
  return sheet;
}

function supervisorRow(q) {
  const points = (q.keypoints || []).map((p) => `${p.decisive ? '[decisive] ' : ''}${p.text}`).join('\n');
  const quotes = (q.supporting_quotes || []).map((x) => x.text).join('\n');
  return { id:q.id, page_type:q.page_type, site:q.site, url:q.url, 'question type':q.type, question:q.question, reference_answer:q.reference_answer, keypoints:points, 'source quotes':quotes, source_location:q.source_location || '', kb_label_final:q.kb_label_final || '' };
}

const qwb = Workbook.create();
addTableSheet(qwb, 'main', questionHeaders, main, { id:25, page_id:24, set:14, page_type:18, site:30, url:48, type:20, question:72, reference_answer:64, source_location:38, keypoints:60, supporting_quotes:70, kb_best_match:55 });
addTableSheet(qwb, 'extra', questionHeaders, extra, { id:25, set:14, page_type:18, site:30, url:48, type:20, question:72, reference_answer:64, keypoints:60, supporting_quotes:70 });
addTableSheet(qwb, 'reserve', questionHeaders, sets.reserve, { id:25, set:14, page_type:18, site:30, url:48, question:72, reference_answer:64, keypoints:60 });
addTableSheet(qwb, 'pending', questionHeaders, sets.pending, { id:25, set:14, page_type:18, site:30, url:48, question:72, reference_answer:64, keypoints:60 });
addTableSheet(qwb, 'removed', removedHeaders, sets.removed, { id:25, set:14, page_type:18, site:30, url:48, question:72, reference_answer:64, keypoints:60, removal_reason:70 });
addTableSheet(qwb, 'manual_checks', manualHeaders, manual, { id:25, kb_check_note:75, table_note:55, note:75 });
addTableSheet(qwb, 'audit', auditHeaders, audit, { id:25, url:45, question:72, reference_answer:64, reason:75, manual_note:75 });
await qwb.recalculate();
const qFile = await SpreadsheetFile.exportXlsx(qwb);
await qFile.save(`${outDir}/questions_v3_final.xlsx`);

const swb = Workbook.create();
addTableSheet(swb, 'main', supervisorHeaders, main.map(supervisorRow), { id:25, page_type:18, site:30, url:50, 'question type':20, question:72, reference_answer:64, keypoints:60, 'source quotes':70, source_location:38 });
addTableSheet(swb, 'extra', supervisorHeaders, extra.map(supervisorRow), { id:25, page_type:18, site:30, url:50, 'question type':20, question:72, reference_answer:64, keypoints:60, 'source quotes':70, source_location:38 });
addTableSheet(swb, 'README', ['Section','Details'], supData.readme, { Section:26, Details:130 });
await swb.recalculate();
const sFile = await SpreadsheetFile.exportXlsx(swb);
await sFile.save(`${outDir}/questions_for_supervisors.xlsx`);

const rw = Workbook.create();
addTableSheet(rw, 'review_sample', sample.headers, sample.rows, { id:25, question:75, reference_answer:70, 'keypoints with quotes':85, url:55, question_ok:18, answer_ok:18, note:55 });
await rw.recalculate();
const rFile = await SpreadsheetFile.exportXlsx(rw);
await rFile.save(`${outDir}/review_sample.xlsx`);

for (const [wb, sheetName, range, path] of [
  [qwb, 'main', 'A1:J8', '/private/tmp/question_set_v3_preview_main.png'],
  [qwb, 'audit', 'A1:N8', '/private/tmp/question_set_v3_preview_audit.png'],
  [swb, 'README', 'A1:B8', '/private/tmp/question_set_v3_preview_readme.png'],
  [rw, 'review_sample', 'A1:H7', '/private/tmp/question_set_v3_preview_sample.png'],
]) {
  const blob = await wb.render({ sheetName, range, scale: 1, format: 'png' });
  await fs.writeFile(path, new Uint8Array(await blob.arrayBuffer()));
}
console.log(JSON.stringify({main:main.length, extra:extra.length, reserve:sets.reserve.length, pending:sets.pending.length, removed:sets.removed.length, audit:audit.length, reviewSample:sample.rows.length, replacedSampleIds:sample.replaced_ids, replacementIds:sample.replacement_ids}));
