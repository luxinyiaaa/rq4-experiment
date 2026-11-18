import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const inputPath = "/private/tmp/s2_main2_review_payload.json";
const outputPath = "outputs/s2/main2/review_sheet_s2_main2.xlsx";
const rows = JSON.parse(await fs.readFile(inputPath, "utf8"));
const headers = rows.length ? Object.keys(rows[0]) : [];
const workbook = Workbook.create();
const sheet = workbook.worksheets.add("S2 Main2 Review");
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
const end = `${lastCol(headers.length)}${matrix.length}`;
sheet.getRange(`A1:${end}`).values = matrix;
sheet.getRange(`A1:${lastCol(headers.length)}1`).format = {
  fill: "#1F4E78",
  font: { name: "Arial", size: 10, bold: true, color: "#FFFFFF" },
  wrapText: true,
  verticalAlignment: "center",
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
sheet.getRange(`A1:${lastCol(headers.length)}1`).format.rowHeight = 34;
if (rows.length) sheet.getRange(`A2:${lastCol(headers.length)}${matrix.length}`).format.rowHeight = 110;
sheet.getRange(`A:${lastCol(headers.length)}`).format.columnWidth = 22;
const customWidths = {
  id: 18, type: 20, user_task: 18, question_user: 42, question_source: 42,
  expected_behaviour: 22, reference_answer: 42,
  "keypoints (decisive, sources, quotes)": 52, "evidence_pages (title, url)": 38,
  quote_verbatim: 30, quote_support: 36, non_decisive_quote_support_failures: 24,
  multi_page: 26, single_page_check: 34, unique_bridge: 30, set_completeness: 36,
  absence_check: 38, conflict_check: 32, condition_semantics: 40, round_trip: 34,
  closed_book: 30, near_duplicate: 30, rules: 36, phrasing_overlap: 22,
  needs_review: 14, needs_review_reason: 34, auto_rejection_reasons: 34, difficulty: 28,
  derivation_check: 34, comparison_multi_validity: 34, validation_retry: 34,
  "decision (accept / edit / reject)": 22, "would_researcher_ask (yes / no)": 26,
  edited_question_user: 38, edited_question_source: 38, edited_reference_answer: 40,
  "edited_keypoints (one per line; [D] = decisive)": 42, note: 32,
};
headers.forEach((header, index) => {
  const column = lastCol(index + 1);
  sheet.getRange(`${column}:${column}`).format.columnWidth = customWidths[header] ?? 22;
});
sheet.freezePanes.freezeRows(1);
sheet.freezePanes.freezeColumns(2);
workbook.recalculate();
const inspection = await workbook.inspect({ kind: "region", sheetId: sheet.name, range: `A1:${lastCol(headers.length)}${Math.min(5, matrix.length)}`, maxChars: 4000 });
console.log(inspection.ndjson);
const preview = await workbook.render({ sheetName: sheet.name, range: "A1:J5", scale: 1, format: "png" });
await fs.writeFile("/private/tmp/s2_main2_review_preview.png", new Uint8Array(await preview.arrayBuffer()));
const file = await SpreadsheetFile.exportXlsx(workbook);
await file.save(outputPath);
console.log(JSON.stringify({ outputPath, rows: rows.length, columns: headers.length }));
