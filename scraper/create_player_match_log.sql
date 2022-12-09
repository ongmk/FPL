DROP TABLE IF EXISTS PLAYER_MATCH_LOG;

CREATE TABLE PLAYER_MATCH_LOG(
    SEASON TEXT,
    PLAYER TEXT,
    DATE TEXT,
    DAY TEXT,
    COMP TEXT,
    ROUND TEXT,
    VENUE TEXT,
    RESULT TEXT,
    SQUAD TEXT,
    OPPONENT TEXT,
    START INTEGER,
    POS TEXT,
    MIN INTEGER,
    GLS INTEGER,
    AST INTEGER,
    PK INTEGER,
    PKATT INTEGER,
    SH INTEGER,
    SOT INTEGER,
    TOUCHES INTEGER,
    XG REAL,
    NPXG REAL,
    XAG REAL,
    SCA INTEGER,
    GCA INTEGER,
    SOTA INTEGER,
    GA INTEGER,
    SAVES INTEGER,
    SAVEPCT INTEGER,
    CS INTEGER,
    PSXG REAL,
    LINK TEXT

);

CREATE UNIQUE INDEX SEASON_PLAYER_DATE_OPPONENT_PK
ON PLAYER_MATCH_LOG(SEASON, PLAYER, DATE, OPPONENT);