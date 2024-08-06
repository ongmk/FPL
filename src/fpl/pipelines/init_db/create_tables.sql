CREATE TABLE IF NOT EXISTS raw_fpl_data(
    season TEXT,
    round REAL,
    element INTEGER,
    full_name TEXT,
    team INTEGER,
    team_name TEXT,
    position TEXT,
    fixture INTEGER,
    starts INTEGER,
    opponent_team INTEGER,
    opponent_team_name TEXT,
    total_points REAL,
    was_home bool,
    kickoff_time TEXT,
    team_h_score REAL,
    team_a_score REAL,
    minutes REAL,
    goals_scored REAL,
    assists REAL,
    clean_sheets REAL,
    goals_conceded REAL,
    own_goals REAL,
    penalties_saved REAL,
    penalties_missed REAL,
    yellow_cards REAL,
    red_cards REAL,
    saves REAL,
    bonus REAL,
    bps REAL,
    influence REAL,
    creativity REAL,
    threat REAL,
    ict_index REAL,
    value REAL,
    transfers_balance REAL,
    selected REAL,
    transfers_in REAL,
    transfers_out REAL,
    expected_goals REAL,
    expected_goal_involvements REAL,
    expected_assists REAL,
    expected_goals_conceded REAL
);

CREATE UNIQUE INDEX IF NOT EXISTS season_element_fixture_pk ON raw_fpl_data(season, element, fixture, round);

----------------------------
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
-- Enable foreign key support in SQLite
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS experiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    keep INTEGER,
    model_best INTEGER,
    start_time TEXT NOT NULL,
    git_message TEXT,
    models TEXT,
    numericalFeatures TEXT,
    categoricalFeatures TEXT
);

----------------------------
CREATE TABLE IF NOT EXISTS inference_results (
    index INTEGER,
    experiment_id INTEGER,
    start_time TEXT NOT NULL,
    PRIMARY KEY (experiment_id, id)
);