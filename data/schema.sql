CREATE TABLE IF NOT EXISTS indicators (
    country_code VARCHAR,
    series_id    VARCHAR,
    date         DATE,
    value        DOUBLE,
    PRIMARY KEY (country_code, series_id, date)
);

CREATE TABLE IF NOT EXISTS model_outputs (
    country_code    VARCHAR,
    model_name      VARCHAR,
    date            DATE,
    recession_prob  DOUBLE,
    inflation_state VARCHAR,
    inflation_probs VARCHAR,
    composite_risk  DOUBLE,
    PRIMARY KEY (country_code, model_name, date)
);
