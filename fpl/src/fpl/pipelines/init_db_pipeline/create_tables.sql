CREATE TABLE raw_team_match_log(
    season TEXT,
    team TEXT,
    date TEXT,
    time TEXT,
    comp TEXT,
    round TEXT,
    day TEXT,
    venue TEXT,
    result TEXT,
    gf INTEGER,
    ga INTEGER,
    opponent TEXT,
    xg REAL,
    xga REAL,
    poss INTEGER,
    link TEXT
);

CREATE UNIQUE INDEX season_team_date_opponent_pk ON raw_team_match_log(season, team, date, opponent);

----------------------------
CREATE TABLE raw_player_match_log(
    season TEXT,
    player TEXT,
    date TEXT,
    day TEXT,
    comp TEXT,
    round TEXT,
    venue TEXT,
    result TEXT,
    squad TEXT,
    opponent TEXT,
    start INTEGER,
    pos TEXT,
    min INTEGER,
    gls INTEGER,
    ast INTEGER,
    pk INTEGER,
    pkatt INTEGER,
    sh INTEGER,
    sot INTEGER,
    touches INTEGER,
    xg REAL,
    npxg REAL,
    xag REAL,
    sca INTEGER,
    gca INTEGER,
    sota INTEGER,
    ga INTEGER,
    saves INTEGER,
    savepct INTEGER,
    cs INTEGER,
    psxg REAL,
    link TEXT
);

CREATE UNIQUE INDEX season_player_date_opponent_pk ON raw_player_match_log(season, player, date, opponent);

----------------------------
CREATE TABLE raw_match_odds(
    season TEXT,
    h_team TEXT,
    a_team TEXT,
    h_score INTEGER,
    a_score INTEGER,
    odds REAL,
    link TEXT
);

CREATE UNIQUE INDEX SEASON_MATCH_SCORE_PK ON raw_match_odds(season, h_team, a_team, h_score, a_score);

----------------------------
-- Enable foreign key support in SQLite
PRAGMA foreign_keys = ON;

CREATE TABLE experiment (
    id INTEGER PRIMARY KEY,
    start_time DATETIME NOT NULL,
    features TEXT NOT NULL
);

----------------------------
CREATE TABLE evaluation_result (
    id INTEGER PRIMARY KEY,
    experiment_id INTEGER,
    start_time DATETIME NOT NULL,
    -- Add more columns for the features here
    FOREIGN KEY (experiment_id, start_time) REFERENCES experiment(id, start_time) ON DELETE CASCADE
);