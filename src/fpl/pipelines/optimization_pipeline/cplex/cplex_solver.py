from pulp import (
    LpStatusOptimal,
    LpStatusInfeasible,
    LpStatusUnbounded,
    LpStatusNotSolved,
)
from .LpRemoteCplex import RemoteCPLEXSolver
import logging
import xml.etree.ElementTree as et

import os
from kedro.config import ConfigLoader

logger = logging.getLogger(__name__)


def get_solution_status(solutionXML):
    solution_header = solutionXML.find("header")
    status_string = solution_header.get("solutionStatusString")
    objective_value_string = solution_header.get("objectiveValue")

    cplex_status = {
        "Optimal": LpStatusOptimal,
        "Feasible": LpStatusOptimal,
        "Infeasible": LpStatusInfeasible,
        "Unbounded": LpStatusUnbounded,
        "Stopped": LpStatusNotSolved,
    }

    status_str = "Undefined"
    if "optimal" in status_string:
        status_str = "Optimal"
    elif "feasible" in status_string:
        status_str = "Feasible"
    elif "infeasible" in status_string:
        status_str = "Infeasible"
    elif "integer unbounded" in status_string:
        status_str = "Integer Unbounded"
    elif "time limit exceeded" in status_string:
        status_str = "Feasible"

    return cplex_status[status_str], status_str, objective_value_string


def solve_lp_with_cplex(
    lp_file_path: str, parameters: dict
) -> tuple[et.Element, float]:
    conf_loader = ConfigLoader(conf_source="conf/")
    cplex_cred = conf_loader["credentials"]["cplex"]
    localpath = os.path.dirname(lp_file_path)
    filename = os.path.splitext(os.path.basename(lp_file_path))[0]

    solver = RemoteCPLEXSolver(
        fileName=filename,
        localPath=localpath,
        config=cplex_cred,
        log=parameters["log"],
        cplexTimeOut=parameters["timeout"],
    )
    cplex_log = solver.solve()
    if cplex_log["infeasible"]:
        raise Exception("Cannot find feasible solution.")
    solution_file_path = os.path.join(localpath, f"{filename}.sol")
    solutionXML = et.parse(solution_file_path).getroot()
    _, status_str, objValString = get_solution_status(solutionXML)
    gap_pct = cplex_log["gap_pct"] if cplex_log else None
    solution_time = cplex_log["solution_time"] if cplex_log else None
    status_str = "Acceptable" if cplex_log and cplex_log["Acceptable"] else status_str
    logger.info(
        f"{status_str} solution found in: {solution_time}s. Gap: {gap_pct}%. Objective: {float(objValString):.2f}"
    )

    return solutionXML, solution_time
