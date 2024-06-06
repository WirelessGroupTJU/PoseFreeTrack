import os,json,argparse
import os.path as osp
from time import sleep
from urllib.request import urlopen
import requests

def tojson(o, ensure_ascii=True):
    return json.dumps(o, default=lambda obj: obj.__dict__, sort_keys=True,ensure_ascii=ensure_ascii)

def save_json(filename, obj, ensure_ascii=True):
    str_json=tojson(obj, ensure_ascii)
    with open(filename,'w') as f:
        f.write(str_json)
        f.close()

def main():
    parser = arg_parser()
    args = parser.parse_args()
    if 'run' == args.action: # 9292
        run_loop(args)
    elif 'add' == args.action: # 9292
        add(args)
    elif 'export' == args.action: # 9292
        export_jobs(args)

def add(args):
    url = 'http://%s/add_jobs' % (args.remotehost)
    response = requests.post(url, files={"upload": open(args.filename, 'rb')})
    print(response.text)

def export_jobs(args):
    url = 'http://%s/export_jobs?id=%s' % (args.remotehost, args.jid)
    outputs_json = get_resp(url, 'export', args.timeout)
    save_json(osp.join('results', '%s.json'%args.jid), outputs_json)

def run_loop(args):
    global remotehost
    global timeout
    global localhost
    global max_retry
    global device
    remotehost = args.remotehost
    timeout = args.timeout
    localhost = args.localhost
    max_retry = args.max_retry
    device = args.device
    jid = args.jid
    stage = 'fetch'
    retry = 0
    while True:
        if stage == 'fetch':
            stage, retry, job_info = fetch(jid)
        elif stage == 'ack':
            stage, retry = ack(job_info, retry)
        elif stage == 'run':
            stage = run(job_info)
        elif stage == 'error':
            stage, retry = error(job_info, retry)
        elif stage == 'finish':
            stage, retry = finish(job_info, retry)
        
        print(stage, retry, len(job_info) if job_info is not None else None)
        if retry:
            print('retry...')
            sleep(3)
        else:
            sleep(0.5)

def fetch(jid):
    if jid is None:
        url = 'http://%s/get_job?server=%s&type=fetch' % (remotehost, localhost)
    else:
        url = 'http://%s/get_job?server=%s&type=fetch&id=%s' % (remotehost, localhost, jid)
    job_info = get_resp(url, 'fetch', timeout)
    if job_info is None:
        return 'fetch', 1, None
    elif 'jid' in job_info:
        print('fetch: ', job_info)
        return 'ack', 0, job_info
    else:
        return 'fetch', 1, job_info

def ack(job_info, retry):
    if retry>max_retry:
        return 'fetch', 0
    url = 'http://%s/get_job?server=%s&type=ack&id=%s&ind=%s' % (remotehost, localhost, job_info['jid'], job_info['ind'])
    info = get_resp(url, 'ack', timeout)
    if info is None or info.get('ack') != 'ok':
        return 'ack', retry+1
    else:
        print('ack: ', job_info)
        return 'run', 0

def run(job_info):
    if job_info['directory'] in os.environ:
        os.chdir(os.environ[job_info['directory']])
    elif osp.exists(job_info['directory']):
        os.chdir(job_info['directory'])
    print('run: ', job_info, ' device %d'%device)
    command = job_info['job']
    if '#' in command:
        command = command[:command.index('#')]
    exit_code = os.system(command+' -dev %d'%device)
    if exit_code:
        return 'error'
    else:
        return 'finish'

def error(job_info, retry):
    if retry>max_retry*2:
        return 'fetch', 0
    url = 'http://%s/get_job?server=%s&type=error&id=%s&ind=%s' % (remotehost, localhost, job_info['jid'], job_info['ind'])
    info = get_resp(url,'error', timeout)
    if info is None or info.get('error') != 'ok':
        return 'error', retry+1
    else:
        print('error: ', job_info)
        return 'fetch', 0

def finish(job_info, retry):
    if retry>max_retry*2:
        return 'fetch', 0
    url = 'http://%s/get_job?server=%s&type=finish&id=%s&ind=%s' % (remotehost, localhost, job_info['jid'], job_info['ind'])
    info = get_resp(url,'finish', timeout)
    if info is None or info.get('finish') != 'ok':
        return 'finish', retry+1
    else:
        print('finish: ', job_info)
        return 'fetch', 0

def get_resp(url, msg_type, timeout):
    try:
        raw_resp = urlopen(url, timeout = timeout)
        resp = raw_resp.read()
        if type(resp) != str:
            resp = resp.decode()
        return json.loads(resp)
    except Exception as e:
        print('Error: %s getting %s from %s' %
            (getattr(e, 'message', str(e)), msg_type, url))
        return None

def arg_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-r', '--remotehost',   type=str,   default='lab.tjunet.top:9293')
    parser.add_argument('-t', '--timeout',      type=int,   default=5)
    parser.add_argument('-m', '--max_retry',    type=int,   default=3)
    subparsers = parser.add_subparsers(dest='action',       help="Action")
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument('-l', '--localhost',type=str,   default='server')
    run_parser.add_argument('-dev', '--device', type=int,   default=0)
    run_parser.add_argument('-i', '--jid',      type=str,   default=None)
    add_parser = subparsers.add_parser("add")
    add_parser.add_argument('-f', '--filename', type=str)
    export_parser = subparsers.add_parser("export")
    export_parser.add_argument('-i', '--jid',   type=str,   default=None)
    return parser

if __name__ == '__main__':
    main()
