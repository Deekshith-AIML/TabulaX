# TabulaX Algorithmic and General Transformation Tests

These two synthetic CSV fixtures supplement `tabulax_test_formats_v2.zip`. The original pack explicitly covers numeric multiplication, uppercase conversion, prefix replacement, deletion and data quality; it does not contain dedicated algorithmic or semantic/general transformation fixtures.

## 1. Algorithmic transformation: `csv/algorithmic_roman_numerals.csv`

Select column `roman_numeral` and provide these input/output examples:

- I -> 1
- IV -> 4
- IX -> 9
- XII -> 12

Ask TabulaX to infer the transformation and apply it to all rows. Expected outputs:

| record_id | roman_numeral | expected output |
|---|---|---:|
| ALG-001 | I | 1 |
| ALG-002 | IV | 4 |
| ALG-003 | IX | 9 |
| ALG-004 | XII | 12 |
| ALG-005 | XIV | 14 |
| ALG-006 | XL | 40 |
| ALG-007 | LXIV | 64 |
| ALG-008 | XC | 90 |
| ALG-009 | CXX | 120 |
| ALG-010 | CD | 400 |

The last six rows are useful holdout checks. If TabulaX exposes the inferred transformation type, record whether it classifies this as `algorithmic`. Classification labels may depend on the implementation; output correctness is the primary check.

## 2. General/semantic transformation: `csv/general_ticket_categories.csv`

Select `ticket_description`. Provide these example mappings:

- Cannot sign in even with the correct password -> Account Access
- Subscription charged twice this month -> Billing
- Laptop shuts down whenever I open it -> Hardware
- Office Wi-Fi disconnects during calls -> Network

Ask TabulaX to categorize the full column using exactly these four labels: `Account Access`, `Billing`, `Hardware`, `Network`.

Expected outputs:

| ticket_id | expected category |
|---|---|
| GEN-001 | Account Access |
| GEN-002 | Account Access |
| GEN-003 | Billing |
| GEN-004 | Billing |
| GEN-005 | Hardware |
| GEN-006 | Hardware |
| GEN-007 | Network |
| GEN-008 | Network |

The examples used to infer the rule are GEN-001, GEN-003, GEN-005 and GEN-007. Check the remaining rows as holdout data. If TabulaX exposes the inferred transformation type, record whether it routes this semantic categorization to `general`/LLM logic.

## Pass criteria

- All rows are preserved and only the selected column changes.
- Output for every row matches the expected result.
- Any generated code or explanation describes a reusable rule, not hardcoded row-specific answers.
- Save/export and reopen the transformed result to check persistence.
- Record the detected transformation type separately from output correctness.

These tests use synthetic data only. They do not prove every possible algorithmic or general transformation will work.
