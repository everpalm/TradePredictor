pipeline {
    agent {
        // Run test on the nodes with the same label
        label 'AMD64_DESKTOP'
    }
    parameters {
        choice(
            choices: [
                'all',
                'steel',
                'electronics',
                'finantial'
            ],
            description: 'Select the test suite to run',
            name: 'MY_SUITE'
        )
    }
    environment {
        MY_PRIVATE_TOKEN = credentials('gitlab-private-token')
        WORK_PATH = 'D:\\workspace\\TradePredictor'
        PYTHONIOENCODING = 'utf-8'
    }
    stages {
        stage('Setup') {
            steps {
                dir("${env.WORK_PATH}") {
                    script {
                        // Install all dependencies, including pytest
                        bat "pipenv sync"
                    }
                }
            }
        }
        stage('Testing') {
            steps {
                dir("${env.WORK_PATH}") {
                    script {
                        if (params.MY_SUITE == 'all') {
                            bat 'pipenv run pytest tests\\'
                        } else if (params.MY_SUITE == 'steel') { 
                            bat 'pipenv run pytest tests\\test_steel'
                        }else if (params.MY_SUITE == 'electronics') {
                            bat 'pipenv run pytest tests\\test_electronics'
                        } else if (params.MY_SUITE == 'financial') {
                            bat 'pipenv run pytest tests\\test_financial'
                        }
                    }
                }
            }
        }
    }
    post {
        always {
            emailext body: 'Test results are available at: $BUILD_URL', subject: 'Test Results', to: 'everpalm@yahoo.com.tw'
        }
        success {
            echo 'Test completed successfully.'
        }
        failure {
            echo 'Test failed.'
        }
    }
}
