from pathlib import Path

import paramiko
from pulp import *
from regex import match
from cplex_connection.config import Config
from logzero import logger

class RemoteCPLEXSolver:
    def __init__(self, fileName, localPath, log=True):
        self.lpFileName = fileName
        self.localPath = localPath
        self.log = log

        self.server = Config.get('cplex', 'server')
        self.remote_path = Config.get('cplex', 'remote_path')
        self.cplex_executable = Config.get('cplex', 'executable')
        self.is_connect_by_pwd = Config.get_boolean('cplex', 'is_connect_by_pwd')
        self.username = Config.get('cplex', 'username')
        if self.is_connect_by_pwd:
            self.password = Config.get('cplex', 'password')
            self.ssh_private_key_file = None
        else:
            self.password = None
            self.ssh_private_key_file = Config.get_purepath('cplex', 'ssh_private_key_file')

    def solve(self):
        # lpProblem.writeLP(self.lpFileName + '.lp')
        _, cplex_log = self.optimizeLPUsingRemoteCPLEX(self.localPath, self.remote_path, self.lpFileName)
        return cplex_log

    def optimizeLPUsingRemoteCPLEX(self, localPath='', remotePath='', lpBaseFileName='', sharableFolder=None):

        success = True
        sshClient = None
        sftpClient = None
        server = self.server
        cplex_executable = self.cplex_executable + "\n"
        is_connect_by_pwd = self.is_connect_by_pwd
        username = self.username
        password = self.password
        ssh_private_key_file = self.ssh_private_key_file
        cplex_log = None

        try:
            sshClient = Helper.openSecureConnectionToRemoteServer(is_connect_by_pwd, server, username, password, ssh_private_key_file)
            remoteTransport = Helper.openSecureTransportToRemoteServer(is_connect_by_pwd, server, username, password, ssh_private_key_file)
            sftpClient = sshClient.open_sftp()

            localLPFileName = localPath + os.sep + lpBaseFileName + '.lp'
            remoteLPFileName = remotePath + '/' + lpBaseFileName + '.lp'
            localSolutionFileName = localPath + os.sep + lpBaseFileName + '.sol'
            remoteSolutionFileName = remotePath + '/' + lpBaseFileName + '.sol'

            # Only copy file to remote server through SFTP incase not using shared folder approach
            if not sharableFolder:
                success = Helper.putFileOnRemoteServer(localFileName=localLPFileName, remotePath=remoteLPFileName, sftp=sftpClient)

            # make backup if solution file exist
            backupFileRemoteChannel = None
            try:
                backupFileRemoteChannel = remoteTransport.open_session()
                backupFileRemoteChannel.exec_command(
                    'mv ' + remoteSolutionFileName + ' ' + remoteSolutionFileName + '.bak')
                backupFileRemoteChannel = None
            except:
                logger.info("Encountered error in backing up solution file...")
                backupFileRemoteChannel = None

            try:
                optimizationChannel = remoteTransport.open_session()

                if success:
                    success, cplex_log = Helper.executeCPLEXThroughSSH(
                        server=server, username=username, password=password,
                        ssh=sshClient, channel=optimizationChannel,
                        remoteLPPath=remoteLPFileName, remoteSolutionPath=remoteSolutionFileName,
                        cplex_executable=cplex_executable,
                        log=self.log)
                optimizationChannel.close()

                if not sharableFolder:
                    if success:
                        success = Helper.getFileFromRemoteServer(localFileName=localSolutionFileName, remotePath=remoteSolutionFileName, sftp=sftpClient)

                    # Delete/cleanup the files on remote server
                    backupFileRemoteChannel = remoteTransport.open_session()
                    backupFileRemoteChannel.exec_command('rm ' + remoteLPFileName)
                    backupFileRemoteChannel = None

                    # Create another channel for solution file as paramiko closes channel after every command
                    backupFileRemoteChannel = remoteTransport.open_session()
                    backupFileRemoteChannel.exec_command('rm ' + remoteSolutionFileName)
                    backupFileRemoteChannel = None

            except Exception as exception:
                logger.info("Encountered exception " + str(exception))
                success = False

            sftpClient.close()
            remoteTransport.close()
            sshClient.close()
            sftpClient = None
            remoteTransport = None
            sshClient = None

        except Exception as exception:
            logger.info(f"Exception encountered in optimizeLPUsingRemoteCPLEX. Exception is {exception}")
            success = False
            try:
                sftpClient.close()
            except:
                pass

            try:
                sshClient.close()
            except:
                pass

        # Return success/failure for this run
        return success, cplex_log


class Helper:

    @staticmethod
    def openSecureConnectionToRemoteServerByPrivateKey(server='', username='', privateKeyFile=''):
        private_key = paramiko.RSAKey.from_private_key_file(privateKeyFile)
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # ssh.load_host_keys(os.path.expanduser(os.path.join("~", ".ssh", "known_hosts")))
        ssh.connect(hostname=server, port=22, username=username, pkey=private_key)
        return ssh

    @staticmethod
    def openSecureConnectionToRemoteServerByPassword(server='', username='', password=''):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        known_hosts_path = os.path.expanduser(os.path.join("~", ".ssh", "known_hosts"))
        if os.path.exists(known_hosts_path):
            ssh.load_host_keys(known_hosts_path)
        else:
            known_hosts_path = "{ROOT_DIR}/files/known_hosts"
            open(known_hosts_path, "a+").close() # create known_hosts if not exists
            ssh.load_host_keys(known_hosts_path)
        ssh.connect(server, username=username, password=password)
        return ssh

    @staticmethod
    def openSecureConnectionToRemoteServer(is_connect_by_pwd=False, server='', username='', password='', ssh_private_key_file=''):
        if is_connect_by_pwd:
            return Helper.openSecureConnectionToRemoteServerByPassword(server, username, password)
        else:
            return Helper.openSecureConnectionToRemoteServerByPrivateKey(server, username, ssh_private_key_file)

    @staticmethod
    def openSecureTransportToRemoteServerByPrivateKey(server='', username='', privateKeyFile=''):
        private_key = paramiko.RSAKey.from_private_key_file(privateKeyFile)
        transport = paramiko.Transport((server, 22))
        transport.connect(username=username, pkey=private_key)
        transport.use_compression()
        transport.window_size = 2147483647
        transport.packetizer.REKEY_BYTES = pow(2, 40) # 1TB max, this is a security degradation
        transport.packetizer.REKEY_PACKETS = pow(2, 40) # 1TB max, this is a security degradation
        return transport

    @staticmethod
    def openSecureTransportToRemoteServerByPassword(server='', username='', password=''):
        transport = paramiko.Transport((server, 22))
        transport.connect(username=username, password=password)
        transport.use_compression()
        transport.window_size = 2147483647
        transport.packetizer.REKEY_BYTES = pow(2, 40)  # 1TB max, this is a security degradation
        transport.packetizer.REKEY_PACKETS = pow(2, 40)  # 1TB max, this is a security degradation
        return transport

    @staticmethod
    def openSecureTransportToRemoteServer(is_connect_by_pwd=False, server='', username='', password='', ssh_private_key_file=''):
        if is_connect_by_pwd:
            return Helper.openSecureTransportToRemoteServerByPassword(server, username, password)
        else:
            return Helper.openSecureTransportToRemoteServerByPrivateKey(server, username, ssh_private_key_file)

    @staticmethod
    def getAbsolutePathFromRelative(relativePath):
        return str(Path(relativePath).resolve())

    @staticmethod
    def putFileOnRemoteServer(localFileName='', remotePath='', sftp=None):
        absoluteLocalFileName = ''
        success = False

        try:
            absoluteLocalFileName = Helper.getAbsolutePathFromRelative(localFileName)
            sftp.put(absoluteLocalFileName, remotePath)

            success = True
        except Exception as e:
            logger.info('Encountered error pushing LP file to Server. Local File:%s, remote file:%s, exception:%s' % (
                absoluteLocalFileName, remotePath, e))
        return success

    @staticmethod
    def getFileFromRemoteServer(localFileName='', remotePath='', sftp=None):
        success = False
        try:
            sftp.get(remotePath, localFileName)
            success = True
        except Exception as exception:
            exceptionMessage = str(exception)
            # Expected for infeasible solution
            if "No such file" not in exceptionMessage:
                logger.info(f'Encountered error retriving solution file from remote Server. Local File: {localFileName}, remote file: {remotePath}, exception: {exceptionMessage}')
                raise
        return success

    @staticmethod
    def getCPLEXTunningParameters():
        cplexTimeOut = 60
        cplexThreads = 0  # set to 0 means using all the avaible threads
        cplex_tunning_cmds = Helper.getCPLEXTunningParametersCommon(cplexTimeOut, cplexThreads)
        return cplex_tunning_cmds

    @staticmethod
    def executeCPLEXThroughSSH(remoteLPPath, remoteSolutionPath, server='', username='', password='',
                               cplex_cmds='', channel=None, ssh=None, cplex_executable='', log=True):
        success = False
        cplex_log = {"timeout": False, "infeasible": False, "gap_pct": 0, "gap_val":0, "Acceptable": False}
        transport = None


        # cplex_executable='/home/oracle/CPLEX_Studio1262/cplex/bin/x86-64_linux/cplex' +"\n"

        cplex_cmds = "read " + remoteLPPath + "\n"
        cplex_cmds += Helper().getCPLEXTunningParameters()
        # for option in self.options:
        #    cplex_cmds += option+"\n"
        '''
        if lp.isMIP():
            if self.mip:
                cplex_cmds += "mipopt\n"
                cplex_cmds += "change problem fixed\n"
            else:
                cplex_cmds += "change problem lp\n"
        '''

        cplex_cmds += "optimize\n"
        cplex_cmds += "write " + remoteSolutionPath + "\n"
        cplex_cmds += "quit\n\n"
        cplex_cmds = cplex_cmds.encode('UTF-8')
        # cplex.communicate(cplex_cmds)
        try:

            stdin, stdout, stderr = ssh.exec_command(cplex_executable)
            stdin.write(cplex_cmds)
            stdin.flush()
            # stdin.flush()
            data = stdout.readlines()
            for line in data:
                if "CPLEX Error  1217: No solution exists" in line:
                    logger.info("executeCPLEXThroughSSH:CPLEX indicated no solution exist....")
                    cplex_log["infeasible"] = True

                if "MIP - Time limit exceeded" in line:
                    logger.info("executeCPLEXThroughSSH:CPLEX time limit exceeded....")
                    cplex_log["timeout"] = True

                objective_line_match = match(r".*Objective = (.*)", line)
                if objective_line_match:
                    cplex_log["objective"] = float(objective_line_match[1].strip())

                solution_line_match = match(r"Solution time = (.*) sec\.  Iterations = (.*)", line)
                if solution_line_match:
                    cplex_log["solution_time"] = float(solution_line_match[1].strip())
                    cplex_log["iteration"] = int(solution_line_match[2].strip().split(' ')[0])

                gap_line_match = match(r"Current MIP best bound = (.*) \(gap = (.*), (.*)%\)", line)
                if gap_line_match:
                    cplex_log["objective"] = float(gap_line_match[1].strip())
                    cplex_log["gap_val"] = float(gap_line_match[2].strip())
                    cplex_log["gap_pct"] = float(gap_line_match[3].strip())
                
                if log:
                    logger.info(line.replace('\n', ''))
            with open("cplex.log", "w", encoding="utf-8") as f:
                f.write("\n".join([l for l in stdout.readlines()]))


            data = stderr.readlines()
            # Examples would be CPLEX executable not found
            if data:
                logger.info("writing stderr output from remote server...")
                for line in data:
                    logger.info(line)

            # Set sucess after all processing completed
            success = True
            if cplex_log["infeasible"]:
                success = False
            if cplex_log["timeout"] and cplex_log["gap_pct"] > 0.02:
                success = False

            if cplex_log["timeout"] and cplex_log["gap_pct"] <= 0.02:
                cplex_log["Acceptable"] = True

        except Exception as exception:
            logger.info("Exception encountered " + str(exception))

        if channel == None:
            try:
                channel.close()
            except:
                pass
            channel = None

            try:
                transport.close()
            except:
                pass
            transport = None

        # Return processing result
        return success, cplex_log

    @staticmethod
    def readsol(filename, logger):
        """Read a CPLEX solution file"""
        import xml.etree.ElementTree as et
        solutionXML = et.parse(filename).getroot()
        solutionheader = solutionXML.find("header")
        statusString = solutionheader.get("solutionStatusString")
        cplexStatus = {
            'Optimal': LpStatusOptimal,
            'Feasible': LpStatusOptimal,
            'Infeasible': LpStatusInfeasible,
            'Unbounded': LpStatusUnbounded,
            'Stopped': LpStatusNotSolved}

        statusstr = 'Undefined'
        if 'optimal' in statusString:
            statusstr = 'Optimal'
        elif 'feasible' in statusString:
            statusstr = 'Feasible'
        elif 'infeasible' in statusString:
            statusstr = 'Infeasible'
        elif 'time limit exceeded' in statusString:
            statusstr = 'Feasible'

        if statusstr not in cplexStatus:
            logger.info(f"Unknown status returned by CPLEX: {statusString}")
            raise PulpSolverError(f"Unknown status returned by CPLEX: {statusString}")
        else:
            logger.info("LP Solve status %s " % statusString)
        status = cplexStatus[statusstr]

        shadowPrices = {}
        slacks = {}
        constraints = solutionXML.find("linearConstraints")
        for constraint in constraints:
            name = constraint.get("name")
            shadowPrice = constraint.get("dual")
            slack = constraint.get("slack")

            if shadowPrice != None:
                shadowPrices[name] = float(shadowPrice)
            else:
                shadowPrices[name] = 0.0

            if slack != None:
                slacks[name] = float(slack)
            else:
                slacks[name] = 0.0

        values = {}
        reducedCosts = {}
        for variable in solutionXML.find("variables"):
            name = variable.get("name")
            value = variable.get("value")
            reducedCost = variable.get("reducedCost")
            if value != None:
                values[name] = float(value)
            else:
                values[name] = 0.0

            if reducedCost != None:
                reducedCosts[name] = float(reducedCost)
            else:
                reducedCosts[name] = 0.0

        return status, values, reducedCosts, shadowPrices, slacks

    @staticmethod
    def getCPLEXTunningParametersCommon(cplexTimeOut, cplexThreads):
        cplex_tunning_cmds = ''
        cplex_tunning_cmds += "set timelimit " + str(cplexTimeOut) + "\n"
        cplex_tunning_cmds += "set threads " + str(cplexThreads) + "\n"
        cplex_tunning_cmds += "set mip limits cutsfactor 30" + "\n"
        cplex_tunning_cmds += "set mip strategy backtrack 0.1" + "\n"
        cplex_tunning_cmds += "set mip strategy heuristicfreq 100" + "\n"
        cplex_tunning_cmds += "set mip strategy startalgorithm 3" + "\n"  # Network Simplex
        cplex_tunning_cmds += "set mip strategy nodeselect 0" + "\n"
        cplex_tunning_cmds += "set mip display 4" + "\n"
        cplex_tunning_cmds += "set mip tolerances mipgap 1e-5" + "\n"
        # cplex_tunning_cmds += "set mip interval 100" + "\n"
        # cplex_tunning_cmds += "set emphasis mip 1" + "\n"
        return cplex_tunning_cmds
