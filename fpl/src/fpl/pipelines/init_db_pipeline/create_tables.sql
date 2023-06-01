CREATE TABLE IF NOT EXISTS raw_team_match_log(
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

CREATE UNIQUE INDEX IF NOT EXISTS season_team_date_opponent_pk ON raw_team_match_log(season, team, date, opponent);

----------------------------
CREATE TABLE IF NOT EXISTS raw_player_match_log(
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

CREATE UNIQUE INDEX IF NOT EXISTS season_player_date_opponent_pk ON raw_player_match_log(season, player, date, opponent);

----------------------------
CREATE TABLE IF NOT EXISTS raw_match_odds(
    season TEXT,
    h_team TEXT,
    a_team TEXT,
    h_score INTEGER,
    a_score INTEGER,
    odds REAL,
    link TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS SEASON_MATCH_SCORE_PK ON raw_match_odds(season, h_team, a_team, h_score, a_score);

----------------------------
-- Enable foreign key support in SQLite
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS experiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keep INTEGER,
    model_best INTEGER,
    start_time TEXT NOT NULL,
    models TEXT,
    numericalFeatures TEXT,
    categoricalFeatures TEXT
);

----------------------------
CREATE TABLE IF NOT EXISTS evaluation_result (
    id INTEGER,
    experiment_id INTEGER,
    start_time TEXT NOT NULL,
    git_message TEXT,
    PRIMARY KEY (experiment_id, id)
);