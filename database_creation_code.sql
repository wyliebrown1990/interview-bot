CREATE TABLE training_data (
    id SERIAL PRIMARY KEY,
    job_title VARCHAR(255) NOT NULL,
    company_name VARCHAR(255) NOT NULL,
    data TEXT NOT NULL,
    embeddings BYTEA,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
