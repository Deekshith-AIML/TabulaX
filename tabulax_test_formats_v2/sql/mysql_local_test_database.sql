-- TabulaX sample data for a LOCAL/TEST MySQL instance only.
-- Review before running. It creates a dedicated database named tabulax_test_formats.
CREATE DATABASE IF NOT EXISTS tabulax_test_formats;
USE tabulax_test_formats;

CREATE TABLE IF NOT EXISTS numeric_transform (
  record_id VARCHAR(10) PRIMARY KEY,
  customer VARCHAR(100) NOT NULL,
  score INT,
  signup_date DATE
);
INSERT INTO numeric_transform (record_id, customer, score, signup_date) VALUES
('N-001','Asha',7,'2025-01-12'),
('N-002','Bharat',8,'2025-01-15'),
('N-003','Charan',6,'2025-02-01'),
('N-004','Divya',5,'2025-02-04'),
('N-005','Esha',7,'2025-02-10')
ON DUPLICATE KEY UPDATE customer=VALUES(customer), score=VALUES(score), signup_date=VALUES(signup_date);

CREATE TABLE IF NOT EXISTS row_delete (
  record_id VARCHAR(10) PRIMARY KEY,
  name VARCHAR(100),
  department VARCHAR(100),
  status VARCHAR(20),
  amount DECIMAL(10,2)
);
INSERT INTO row_delete (record_id,name,department,status,amount) VALUES
('R-001','Asha','Sales','Active',1200.00),
('R-002','Bharat','Engineering','Active',1800.00),
('R-003','Charan','Sales','Inactive',1200.00),
('R-004','Divya','Finance','Active',2500.00)
ON DUPLICATE KEY UPDATE name=VALUES(name), department=VALUES(department), status=VALUES(status), amount=VALUES(amount);

CREATE TABLE IF NOT EXISTS quality_edge_cases (
  row_id INT PRIMARY KEY,
  label VARCHAR(100),
  amount VARCHAR(40),
  event_date VARCHAR(40),
  email VARCHAR(120),
  notes VARCHAR(200)
);
INSERT INTO quality_edge_cases (row_id,label,amount,event_date,email,notes) VALUES
(1,'Normal','10.50','2026-01-12','a@example.com','valid row'),
(2,'Blank amount','','2026-01-13','b@example.com','missing amount'),
(3,'Comma value','1,200.00','2026/01/14','','comma in amount and missing email'),
(4,'Bad date','0','not-a-date','invalid-email','malformed values')
ON DUPLICATE KEY UPDATE label=VALUES(label), amount=VALUES(amount), event_date=VALUES(event_date), email=VALUES(email), notes=VALUES(notes);
