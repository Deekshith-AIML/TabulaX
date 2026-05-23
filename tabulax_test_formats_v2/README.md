# TabulaX Expanded Sample Test Pack

Synthetic datasets for checking TabulaX import, transformations, data quality, table editing, history, download and database connector behavior.

## Supported-format test files
- `csv/`: CSV samples for numeric transformations, string transformations, messy data, row deletion, column deletion and inventory.
- `xlsx/tabulax_test_workbook.xlsx`: six worksheets covering the same scenarios.
- `json/`: flat object arrays plus a nested JSON file with nulls and `_id` values.
- `sql/mysql_local_test_database.sql`: local/test MySQL schema and sample tables only.

## Optional parser probes
- `optional_formats/numeric_transform.tsv`
- `optional_formats/numeric_transform.xml`

TSV/XML are included as parser probes only. They are not assumed to be supported by TabulaX unless the current importer explicitly supports them.

## Try these transformations
### Multiply by ten
Dataset: `numeric_transform.csv` or matching JSON/workbook sheet. Select `score` and enter:
```
7 → 70
8 → 80
```
For values 7, 8, 6, 5, and 7, the simple multiplication-by-ten rule predicts 70, 80, 60, 50, and 70. These examples do not imply 6→61 or 5→54.

### Uppercase names
Dataset: `string_transform.csv`. Select `first_name`:
```
ananya → ANANYA
kiran → KIRAN
```

### Prefix replacement
Dataset: `inventory.csv`. Select `sku`:
```
SKU-001 → ITEM-001
SKU-002 → ITEM-002
```

## Deletion checks
- `row_delete.csv`: delete R-002 and R-004; verify other rows remain. R-001 and R-005 intentionally contain duplicate-looking data but have distinct IDs.
- `column_delete.csv`: delete `salary` or `notes`; verify remaining column order and values.
- Use history/rollback to verify that the saved version can be opened, downloaded and restored according to the application's implementation.

## Data quality checks
Use `messy_data_quality.csv` and `nested_objects_nulls.json` to test blank values, nulls, commas inside quoted CSV fields, invalid dates, invalid email-like values, nested structures, booleans, and numeric values.

## SQL safety
`mysql_local_test_database.sql` creates the dedicated database `tabulax_test_formats`. Review the script and run it only against a disposable local/test MySQL server. Do not use production credentials/data.

## Notes
- All data is synthetic.
- Do not run destructive deletion or rollback tests against production databases.
- A parser fixture does not imply the file format is supported by the current application.
