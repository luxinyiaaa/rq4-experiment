import fs from 'node:fs/promises';
import { SpreadsheetFile, Workbook } from '@oai/artifact-tool';

const [dataPath, outputPath] = process.argv.slice(2);
if (!dataPath || !outputPath) throw new Error('Usage: node expert_api_full_summary_workbook.mjs DATA.json OUTPUT.xlsx');
const data = JSON.parse(await fs.readFile(dataPath, 'utf8'));
const headers = ['id','set','type','page_type','question','reference_answer','source_url','openai','openai_gpt56_sol','openai_gpt6_astra','egi'];
const rows = [headers, ...data.rows.map((row) => headers.map((header) => row[header] ?? ''))];
const workbook = Workbook.create();
const sheet = workbook.worksheets.add('answers');
sheet.getRangeByIndexes(0, 0, rows.length, headers.length).values = rows;
sheet.showGridLines = false;
sheet.freezePanes.freezeRows(1);
sheet.getRangeByIndexes(0, 0, 1, headers.length).format = {
  fill: '#17365D', font: { bold: true, color: '#FFFFFF' }, wrapText: true, verticalAlignment: 'center'
};
sheet.getRangeByIndexes(0, 0, rows.length, headers.length).format.wrapText = true;
sheet.getRangeByIndexes(0, 0, rows.length, headers.length).format.verticalAlignment = 'top';
sheet.getRangeByIndexes(0, 0, 1, headers.length).format.rowHeight = 32;
for (const [col, width] of Object.entries({ id:25, set:14, type:20, page_type:18, question:72, reference_answer:64, source_url:52, openai:72, openai_gpt56_sol:72, openai_gpt6_astra:72, egi:72 })) {
  const index = headers.indexOf(col);
  sheet.getRangeByIndexes(0, index, rows.length, 1).format.columnWidth = width;
}
await workbook.recalculate();
const file = await SpreadsheetFile.exportXlsx(workbook);
await file.save(outputPath);
