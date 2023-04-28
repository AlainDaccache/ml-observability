
CREATE TABLE IF NOT EXISTS ML_PREDICTIONS (
    ID          SERIAL          PRIMARY KEY,
    EVENT_TIME  TIMESTAMP       NOT NULL,
    HOME_ID     VARCHAR(50)     NOT NULL,
    PREDICTION  DECIMAL         NOT NULL,
    MODEL_USED  VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS BOSTON_DATASET (
    HOME_ID     VARCHAR(50) PRIMARY KEY,
    CRIM        FLOAT,
    ZN          FLOAT,
    INDUS       FLOAT,
    CHAS        FLOAT,
    NOX         FLOAT,
    RM          FLOAT,
    AGE         FLOAT,
    DIS         FLOAT,
    RAD         FLOAT,
    TAX         FLOAT,
    PTRATIO     FLOAT,
    B           FLOAT,
    LSTAT       FLOAT,
    MEDV        FLOAT
);