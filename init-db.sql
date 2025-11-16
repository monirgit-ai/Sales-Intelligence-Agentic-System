-- Initialize database for Sales Intelligence Agentic System

-- Create database if not exists (handled by POSTGRES_DB env var)
-- CREATE DATABASE sales_intelligence;

-- Create table for storing sales intelligence reports
CREATE TABLE IF NOT EXISTS sales_intelligence_reports (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    narrative_report TEXT,
    execution_status VARCHAR(50),
    has_actions BOOLEAN DEFAULT FALSE,
    actions_count INTEGER DEFAULT 0,
    full_results JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_timestamp ON sales_intelligence_reports(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_execution_status ON sales_intelligence_reports(execution_status);
CREATE INDEX IF NOT EXISTS idx_has_actions ON sales_intelligence_reports(has_actions);

-- Create view for latest reports
CREATE OR REPLACE VIEW latest_reports AS
SELECT 
    id,
    timestamp,
    execution_status,
    has_actions,
    actions_count,
    LEFT(narrative_report, 200) as narrative_preview,
    created_at
FROM sales_intelligence_reports
ORDER BY timestamp DESC
LIMIT 50;

-- Grant permissions (adjust as needed)
-- GRANT ALL PRIVILEGES ON TABLE sales_intelligence_reports TO n8n;
-- GRANT USAGE, SELECT ON SEQUENCE sales_intelligence_reports_id_seq TO n8n;

