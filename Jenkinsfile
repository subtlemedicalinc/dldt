node {
    // platform Vars
    def ENV = ""
    def AWS_REGION = "us-east-1"
    def GIT_CREDS_ID = ""
    def PACKAGE = true
    def DEPLOY_BUCKET = ""

    stage("Platform Env Setup") {

        if (env.BRANCH_NAME ==~ /(master)/) {
            ENV = "prod"
            AWS_REGION = "us-west-1"
        } else if (env.BRANCH_NAME ==~ /(hotfix\/(.*)|release\/(.*))/) {
            ENV = "stage"
            // AWS region is different only for the app artifact bucket in stage
            AWS_REGION = "us-west-2"
        } else if (env.BRANCH_NAME ==~ /develop/) {
            ENV = "dev"
        } else {
            ENV = "dev"
            PACKAGE = false
        }

        DEPLOY_BUCKET = "com-subtlemedical-${ENV}-public"
        GIT_CREDS_ID = env.GIT_CREDS_ID

        dir('subtle-platform-utils') {
            git(
                url: 'https://github.com/subtlemedicalinc/subtle-platform-utils.git',
                credentialsId: GIT_CREDS_ID,
                branch: "master"
            )
        }
    }

    stage("Checkout") {
        checkout scm

        /* get version */
        if (env.BRANCH_NAME ==~ /(master|release\/(.*)|hotfix\/(.*))/) {
            def branch_name = env.BRANCH_NAME
            sh 'git log -n 1 ${branch_name} --pretty=format:"%H" > GIT_COMMIT'
            GIT_COMMIT = readFile('GIT_COMMIT').toString()
        }
    }

    stage("Build") {
        echo 'Building...'
        docker.image("centos:7").inside {
            sh '''
                yum -y install yum-utils zip unzip > /dev/null
                yum -y groupinstall development > /dev/null
                yum -y install https://centos7.iuscommunity.org/ius-release.rpm > /dev/null
                yum -y install python35u python35u-pip python35u-devel cmake3 > /dev/null
                yum -y install boost-devel > /dev/null
                git submodule init
                git submodule update --recursive
                pip3.5 install cython > /dev/null
                python3.5 com-subtlemedical-dldt/build_subtle.py -c cmake3
            '''
        }
    }

    stage("Package and Upload") {
        if (env.BRANCH_NAME ==~ /(master|release\/(.*)|hotfix\/(.*))/) {
            echo "Uploading the artifacts to S3..."
            s3Upload(
                file: "com-subtlemedical-dldt/inference-engine.zip",
                bucket: "${DEPLOY_BUCKET}",
                path: "dldt/${GIT_COMMIT}/inference-engine.zip",
                acl: "PublicRead"
            )
            s3Upload(
                file: "com-subtlemedical-dldt/model-optimizer.zip",
                bucket: "${DEPLOY_BUCKET}",
                path: "dldt/${GIT_COMMIT}/model-optimizer.zip",
                acl: "PublicRead"
            )
        } else {
            echo "Skipping deployment for feature branches"
        }
    }

}