import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const inputPath = "outputs/s2/main3/review_payload.json";
const outputPath = "outputs/s2/main3/review_sheet_s2_final.xlsx";
const rows = JSON.parse(await fs.readFile(inputPath, "utf8"));
const headers = rows.length ? Object.keys(rows[0]) : [];
const workbook = Workbook.create();
const sheet = workbook.worksheets.add("S2 Final Review");
sheet.showGridLines = false;
const matrix = [headers, ...rows.map((row) => headers.map((header) => row[header] ?? ""))];
const lastCol = (number) => {
  let result = "";
  while (number > 0) {
    const remainder = (number - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    number = Math.floor((number - 1) / 26);
  }
  return result;
};
const endCol = lastCol(headers.length);
const end = `${endCol}${matrix.length}`;
sheet.getRange(`A1:${end}`).values = matrix;
sheet.getRange(`A1:${endCol}1`).format = {
  fill: "#1F4E78",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  wrapText: true,
  verticalAlignment: "center",
  horizontalAlignment: "center",
};
sheet.getRange(`A2:${end}`).format = {
  font: { name: "Arial", size: 10, color: "#1F1F1F" },
  wrapText: true,
  verticalAlignment: "top",
};
sheet.getRange(`A1:${end}`).format.borders = {
  preset: "insideHorizontal",
  style: "thin",
  color: "#D9E2F3",
};
sheet.getRange(`A1:${endCol}1`).format.rowHeight = 36;
if (rows.length) sheet.getRange(`A2:${end}`).format.rowHeight = 110;
sheet.getRange(`A:${endCol}`).format.columnWidth = 22;
const customWidths = {
  id: 22, type: 20, user_task: 18, question_user: 42, question_source: 42,
  expected_behaviour: 22, reference_answer: 42,
  "keypoints (decisive, sources, quotes)": 52, "evidence_pages (title, url)": 40,
  quote_verbatim: 32, quote_support: 38, non_decisive_quote_support_failures: 26,
  multi_page: 26, single_page_check: 36, unique_bridge: 32, set_completeness: 40,
  absence_check: 40, conflict_check: 34, condition_semantics: 42, round_trip: 36,
  closed_book: 32, near_duplicate: 32, rules: 38, phrasing_overlap: 24,
  needs_review: 14, needs_review_reason: 36, auto_rejection_reasons: 36, difficulty: 28,
  derivation_check: 36, comparison_multi_validity: 36, validation_retry: 36,
  "decision (accept / edit / reject)": 24, "would_researcher_ask (yes / no)": 28,
  edited_question_user: 40, edited_question_source: 40, edited_reference_answer: 42,
  "edited_keypoints (one per line; [D] = decisive)": 44, note: 34,
};
headers.forEach((header, index) => {
  const column = lastCol(index + 1);
  sheet.getRange(`${column}:${column}`).format.columnWidth = customWidths[header] ?? 22;
});
sheet.freezePanes.freezeRows(1);
sheet.freezePanes.freezeColumns(2);
workbook.recalculate();
const inspection = await workbook.inspect({ kind: "region", sheetId: sheet.name,
  range: `A1:${lastCol(Math.min(headers.length, 10))}${Math.min(5, matrix.length)}`, maxChars: 4000 });
console.log(inspection.ndjson);
const preview = await workbook.render({ sheetName: sheet.name, range: "A1:J5", scale: 1, format: "png" });
await fs.writeFile("/private/tmp/s2_main3_review_preview.png", new Uint8Array(await preview.arrayBuffer()));
const file = await SpreadsheetFile.exportXlsx(workbook);
await file.save(outputPath);
console.log(JSON.stringify({ outputPath, rows: rows.length, columns: headers.length }));
