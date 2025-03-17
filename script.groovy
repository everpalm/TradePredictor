def safe_bat(String command) {
    try {
        bat command
    } catch (Exception e) {
        echo "Command failed: ${command}"
        error "Stopping pipeline due to failure."
    }
}

def sync_env() {
    safe_bat 'pipenv sync'
}

def run_test(String path, String key = '', String additional_args = '') {
    safe_sh "pipenv run pytest '${path}' --private_token='${key}' ${additional_args}"
}


def test_pep8(String path) {
    safe_sh "pipenv run pylint '${path}' --fail-under=7.0"
}

def test_sanity(String path) {
    run_test(path, '', "--testmon --cov=${path} --json-report")
}

def test_regression(String path) {
    run_test(path, '', "--cov=${path} --json-report")
}

def test_unit(String key) {
    run_test(tests\\test_unit, key, "--cov=${path} --json-report")
}

return this
