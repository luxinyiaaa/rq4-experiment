import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = process.cwd();
const outDir = path.join(root, "outputs/rawpage_questions/question_set_v2");
const readJsonl = async (file) => (await fs.readFile(file, "utf8")).trim().split(/\n/).filter(Boolean).map((line) => JSON.parse(line));
const questions = await readJsonl(path.join(outDir, "questions_all.jsonl"));
const removed = await readJsonl(path.join(root, "outputs/rawpage_questions/question_set_final/removed_questions.jsonl"));

const columns = [
  "id", "page_id", "set", "page_type", "site", "url", "type", "type_substituted",
  "question", "reference_answer", "source_location", "keypoints", "supporting_quotes",
  "false_premise", "kb_label_final", "needs_kb_check", "quote_in_page", "in_kb",
  "in_kb_any", "kb_other_urls", "kb_best_match", "similarity_label", "live_in_kb",
  "table_based", "quote_cut_off", "duplicate", "duplicate_of", "teacher_model",
];
const removedColumns = ["id", "page_type", "site", "url", "type", "question", "reference_answer", "reason_code", "removal_reason"];
const excelValue = (value) => {
  if (value === undefined || value === null) return "";
  if (Array.isArray(value)) return value.map((item) => typeof item === "object" ? JSON.stringify(item) : String(item)).join("\n");
  if (typeof value === "object") return JSON.stringify(value);
  const text = String(value);
  return text.startsWith("=") ? `'${text}` : text;
};
const keypoints = (row) => (row.keypoints || []).map((point) => `${point.decisive ? "[decisive] " : "[non-decisive] "}${point.text}`).join("\n");
const quotes = (row) => (row.supporting_quotes || []).map((quote) => `[${quote.source || "passage"}] ${quote.text || ""}`).join("\n");
function project(row, cols) {
  return cols.map((col) => {
    if (col === "keypoints") return keypoints(row);
    if (col === "supporting_quotes") return quotes(row);
    return excelValue(row[col]);
  });
}
function colName(n) {
  let value = n;
  let result = "";
  while (value > 0) {
    const rem = (value - 1) % 26;
    result = String.fromCharCode(65 + rem) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
}
function addSheet(workbook, name, headers, records, extra = []) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  const data = records.map((row) => project(row, headers));
  const matrix = [headers, ...data.map((row, index) => [...row, ...extra.map((field) => field === "manual_kb_label" || field === "note" || field === "answer_ok" ? "" : excelValue(records[index][field]))])];
  const range = sheet.getRangeByIndexes(0, 0, matrix.length, matrix[0].length);
  range.values = matrix;
  range.format.font = { name: "Aptos", size: 10, color: "#222222" };
  range.format.wrapText = true;
  range.format.verticalAlignment = "top";
  const head = sheet.getRangeByIndexes(0, 0, 1, matrix[0].length);
  head.format.fill = "#1F4E78";
  head.format.font = { name: "Aptos", size: 10, bold: true, color: "#FFFFFF" };
  head.format.rowHeight = 30;
  sheet.freezePanes.freezeRows(1);
  const widths = matrix[0].map((header) => ({
    id: 24, page_id: 24, set: 14, page_type: 18, site: 24, url: 48, type: 18,
    type_substituted: 18, question: 56, reference_answer: 56, source_location: 38,
    keypoints: 72, supporting_quotes: 80, false_premise: 56, kb_label_final: 20,
    needs_kb_check: 18, quote_in_page: 16, in_kb: 14, in_kb_any: 16,
    kb_other_urls: 52, kb_best_match: 65, similarity_label: 20, live_in_kb: 16,
    table_based: 16, quote_cut_off: 16, duplicate: 14, duplicate_of: 24,
    teacher_model: 28, manual_kb_label: 22, note: 50, answer_ok: 16,
    reason_code: 24, removal_reason: 56,
  }[header] || 20));
  widths.forEach((width, col) => { sheet.getRangeByIndexes(0, col, matrix.length, 1).format.columnWidth = width; });
  if (matrix.length > 1) sheet.tables.add(`A1:${colName(matrix[0].length)}${matrix.length}`, true, `${name.replace(/[^A-Za-z0-9]/g, "")}Table`);
  return sheet;
}

const main = questions.filter((row) => row.set === "main");
const extraRows = questions.filter((row) => row.set === "extra");
const pending = questions.filter((row) => row.set === "pending");
const newQuestions = questions.filter((row) => row.v2_added === true);
const tableChecks = newQuestions.filter((row) => row.table_based === "yes");
const workbook = Workbook.create();
addSheet(workbook, "main", columns, main);
addSheet(workbook, "extra", columns, extraRows);
addSheet(workbook, "pending_kb_check", [...columns, "manual_kb_label", "note"], pending);
addSheet(workbook, "table_check_new", [...columns, "answer_ok", "note"], tableChecks);
addSheet(workbook, "removed", removedColumns, removed);
await (await SpreadsheetFile.exportXlsx(workbook)).save(path.join(outDir, "questions_v2.xlsx"));
console.log(JSON.stringify({ main: main.length, extra: extraRows.length, pending: pending.length, newTableChecks: tableChecks.length, removed: removed.length }));
