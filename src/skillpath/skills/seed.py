"""Hand-curated skill taxonomy: canonical skill names, aliases and categories.

This is the source of truth. Edit it here, then rebuild data/skills.csv with
    python -m skillpath.skills.build

Format, one skill per line:  Canonical Name | alias, alias, ...
- Aliases are matched after normalization (lowercase, collapsed whitespace, trailing
  "skills" removed; see taxonomy.skill_key), so case and spacing variants are not needed.
- Each alias belongs to exactly one skill. The build fails if two skills share an alias.

EXACT_ONLY lists names and aliases that are fine in a skill list but ambiguous in prose
("R", "Go", "AD", "rest"). They match whole skill-list entries only, never free text (M1-07).
"""

EXACT_ONLY = {
    # short or common-word forms of skills
    "r", "go", "c", "dart", "julia", "swift", "lua", "chef", "puppet", "express", "rails", "spark",
    "hive", "shell", "lambda", "exchange", "rest", "api", "apis", "ad", "tf", "ts", "js", "vb", "bi",
    "rl", "ml", "ai", "qa", "ui", "ux", "a+", "elk", "pci", "san", "nas", "soc", "sso", "iam", "erp", "crm",
    "f5", "wan", "lan", "uat", "tdd", "bdd", "tpm", "dba", "mdm", "adf", "oci", "ood", "oop", "dsa", "cli", "os",
    # generic words that only mean a skill when listed as one
    "analysis", "analytics", "organization", "initiative", "requirements", "training", "programming",
    "coding", "testing", "security", "cloud", "networks", "wireless", "hardware", "storage", "backup",
    "embedded", "routing", "switching", "databases", "database", "visualization", "optimization",
    "automation", "monitoring", "documentation", "collaboration", "flexibility", "presentations",
    "communication", "leadership", "coaching", "interpersonal", "analytical", "adaptable", "terminal",
    "frontend", "backend", "front end", "back end", "full stack", "transformers", "forecasting",
    "experimentation", "excel", "oracle", "apache",
}

SEED = {
    "Programming Languages": """
        Python | python3, python programming, python scripting
        Java | java programming, core java, java se, java ee, j2ee
        JavaScript | js, javascript programming, ecmascript, es6
        TypeScript | ts
        C++ | cpp, c plus plus, c/c++
        C# | c sharp, csharp
        C | c programming, c language
        Go | golang, go programming, go language
        Rust | rust programming
        Kotlin |
        Swift | swift programming
        Objective-C | objective c, objc
        Scala |
        Ruby | ruby programming
        PHP |
        Perl |
        R | r programming, r language, rstudio
        MATLAB | the mathworks matlab
        SQL | structured query language, structured query language sql, sql programming, sql queries, t-sql, tsql, pl/sql, plsql
        Bash | bash scripting, shell scripting, shell script, shell, unix shell
        PowerShell | powershell scripting, windows powershell
        Visual Basic | vb, vba, vb.net, microsoft visual basic, microsoft visual basic for applications vba, visual basic for applications
        Dart |
        Elixir |
        Haskell |
        Julia |
        Lua |
        Groovy |
        COBOL |
        Fortran |
        Assembly | assembly language
        Solidity |
    """,
    "Web Frontend": """
        HTML | html5, hypertext markup language, hypertext markup language html
        CSS | css3, cascading style sheets
        React | reactjs, react.js, react js
        Angular | angularjs, angular.js, angular js
        Vue.js | vue, vuejs, vue js
        Next.js | nextjs, next js
        Svelte |
        jQuery | jquery
        Redux |
        Tailwind CSS | tailwind
        Bootstrap |
        Sass | scss
        Webpack |
        Frontend Development | front end development, front-end development, frontend, front end, front-end, ui development
        Responsive Design | responsive web design
        Web Development | web application development, web applications, web technologies
        Accessibility | web accessibility, wcag, 508 compliance
    """,
    "Backend & APIs": """
        Node.js | node, nodejs, node js
        Express.js | express, expressjs
        Django |
        Flask |
        FastAPI | fast api
        Spring | spring framework, spring boot, springboot, spring mvc
        .NET | dotnet, .net framework, .net core, asp.net, asp.net core, microsoft .net framework
        Ruby on Rails | rails, ror
        Laravel |
        GraphQL |
        REST APIs | rest, rest api, restful, restful apis, restful api, restful web services, rest services, api, apis, api development, web services
        gRPC | grpc
        SOAP | soap web services, soap apis, wsdl
        Microservices | microservices architecture, microservice, micro services
        Backend Development | back end development, back-end development, backend, back end, back-end, server side development
        Full Stack Development | full stack, fullstack, full-stack, full stack development, full-stack development
        Message Queues | message queue, message queuing, rabbitmq, activemq, amazon sqs, sqs
        Apache Kafka | kafka
        Hibernate |
        Object-Oriented Programming | oop, object oriented programming, object oriented design, ood, object-oriented design
        Design Patterns | software design patterns
        Data Structures | data structures and algorithms, algorithms, dsa
        System Design | software architecture, systems architecture, system architecture, distributed systems, scalable systems, scalability
    """,
    "Mobile": """
        iOS Development | ios, ios development, iphone development
        Android Development | android, android development, android sdk
        React Native |
        Flutter |
        SwiftUI |
        Xamarin |
        Mobile Development | mobile app development, mobile applications, mobile application development
    """,
    "Databases": """
        PostgreSQL | postgres, postgre sql
        MySQL | my sql
        Microsoft SQL Server | sql server, mssql, ms sql, ms sql server
        Oracle Database | oracle, oracle db, oracle sql, oracle database
        SQLite |
        MongoDB | mongo
        Cassandra | apache cassandra
        Redis |
        DynamoDB | amazon dynamodb, aws dynamodb
        Elasticsearch | elastic search, elk, elk stack, elastic stack, opensearch
        Neo4j | graph databases, graph database
        NoSQL | nosql databases, nosql database
        Relational Databases | rdbms, relational database, relational database management systems, databases, database
        Database Design | database management, database development, database architecture, schema design
        Database Administration | dba, database administrator
        Vector Databases | vector database, vector db, vector search, pinecone, chromadb, faiss, weaviate, milvus, pgvector
    """,
    "Cloud": """
        AWS | amazon web services, amazon web services aws, amazon web services aws software, aws cloud
        Microsoft Azure | azure, azure cloud, microsoft azure software
        Google Cloud Platform | gcp, google cloud, google cloud platform gcp
        AWS Lambda | lambda
        Amazon EC2 | ec2, aws ec2
        Amazon S3 | s3, aws s3
        Amazon Redshift | redshift
        Serverless | serverless computing, serverless architecture
        Cloud Computing | cloud, cloud services, cloud technologies, cloud platforms, cloud infrastructure, cloud solutions, saas, paas, iaas
        Cloud Architecture | cloud design, cloud solutions architecture
        Cloud Security | cloud security architecture
        Cloud Migration |
        Azure DevOps | azure devops services, vsts
        OpenStack |
        IBM Cloud |
        Oracle Cloud | oci, oracle cloud infrastructure
    """,
    "DevOps & Infrastructure": """
        Docker | docker containers
        Kubernetes | k8s, kubernetes administration, eks, aks, gke, openshift
        Containers | containerization, container orchestration
        Terraform | hashicorp terraform
        Ansible |
        Puppet |
        Chef |
        CloudFormation | aws cloudformation
        Infrastructure as Code | iac, infrastructure-as-code
        CI/CD | ci/cd pipelines, cicd, ci cd, continuous integration, continuous delivery, continuous deployment, continuous integration/continuous deployment, ci/cd pipeline
        Jenkins |
        GitHub Actions |
        GitLab CI | gitlab ci/cd
        DevOps | devops practices, devops tools
        Site Reliability Engineering | sre
        Monitoring | system monitoring, observability, application monitoring, performance monitoring
        Prometheus |
        Grafana |
        Datadog |
        Splunk | splunk enterprise
        New Relic |
        Nginx |
        Apache HTTP Server | apache, apache web server
        Helm |
        Argo CD | argocd
        Linux Administration | linux system administration, linux administration, linux server administration
        Release Management | release engineering, build and release
        Configuration Management |
        Scripting | scripting languages, automation scripting
        Automation | process automation, it automation
    """,
    "Data Engineering": """
        Data Engineering | data engineer
        ETL | etl processes, etl development, elt, extract transform load, data pipelines, data pipeline, etl pipelines
        Apache Spark | spark, pyspark, spark sql
        Hadoop | apache hadoop, hdfs, mapreduce
        Hive | apache hive
        Apache Airflow | airflow
        dbt | data build tool
        Databricks |
        Snowflake |
        Data Warehousing | data warehouse, data warehouses, enterprise data warehouse, dwh
        Data Lakes | data lake, data lakehouse, lakehouse
        Data Modeling | data modelling, dimensional modeling, data models
        Big Data | big data technologies, big data analytics
        Data Integration |
        Data Governance | data management, master data management, mdm, data stewardship
        Data Quality | data integrity, data validation, data cleansing, data cleaning
        Stream Processing | streaming data, real-time data processing, apache flink, flink, kinesis, amazon kinesis
        Informatica |
        SSIS | sql server integration services
        Talend |
        Azure Data Factory | adf
        BigQuery | google bigquery
        Teradata | teradata database
        Data Architecture |
        Data Migration |
    """,
    "Data Analysis & BI": """
        Data Analysis | data analytics, analytics, data analyst
        Data Visualization | data visualisation, visualization, dashboards, dashboard development
        Tableau |
        Power BI | powerbi, microsoft power bi
        Looker |
        Qlik | qlikview, qlik sense
        Microsoft Excel | excel, ms excel, advanced excel, excel vba, spreadsheets
        Business Intelligence | bi, business intelligence tools
        Statistics | statistical analysis, statistical modeling, statistical methods, applied statistics
        A/B Testing | ab testing, experimentation, experimental design
        SAS |
        SPSS | ibm spss statistics
        Pandas |
        NumPy | numpy
        Jupyter | jupyter notebooks, jupyter notebook
        Reporting | data reporting, report development, ssrs, sql server reporting services
        Google Analytics |
        Predictive Modeling | predictive analytics
        Quantitative Analysis | quantitative research
    """,
    "Machine Learning & AI": """
        Machine Learning | ml, machine learning algorithms, machine learning models
        Deep Learning | neural networks, deep neural networks, dnn
        Artificial Intelligence | ai, ai/ml, ml/ai
        Natural Language Processing | nlp, natural language understanding, text mining, text analytics
        Computer Vision | image processing, image recognition, object detection
        Generative AI | genai, gen ai, generative ai models
        Large Language Models | llm, llms, large language model
        Prompt Engineering |
        Retrieval-Augmented Generation | rag, retrieval augmented generation
        LangChain |
        LlamaIndex | llama index
        LangGraph |
        AI Agents | agentic ai, autonomous agents, multi-agent systems
        OpenAI API | openai, gpt, chatgpt, gpt-4, azure openai
        Hugging Face | huggingface, transformers
        TensorFlow | tensorflow 2, tf
        PyTorch | torch
        Keras |
        scikit-learn | sklearn, scikit learn
        XGBoost | xgboost, lightgbm, gradient boosting
        MLOps | ml ops, machine learning operations, model deployment, mlflow, kubeflow
        Reinforcement Learning | rl
        Recommender Systems | recommendation systems, recommendation engines, collaborative filtering
        Feature Engineering |
        Model Evaluation | model validation, model monitoring
        Time Series Analysis | time series, forecasting, time series forecasting
        Data Science | data scientist
        Data Mining |
        Embeddings | vector embeddings, word embeddings
        Amazon SageMaker | sagemaker, aws sagemaker
        Azure Machine Learning | azure ml, azure ai, azure ai foundry, azure cognitive services
        Vertex AI | google vertex ai
        CUDA | gpu programming
    """,
    "Security": """
        Cybersecurity | cyber security, information security, infosec, it security, security, data security
        Network Security | firewalls, firewall, firewall management, ids/ips, intrusion detection
        Application Security | appsec, secure coding, owasp, application security testing
        Penetration Testing | pen testing, pentesting, ethical hacking
        Vulnerability Management | vulnerability assessment, vulnerability scanning, vulnerability assessments
        Identity and Access Management | iam, identity management, access management, access control, okta, sso, single sign-on
        SIEM | security information and event management
        Incident Response | security incident response, incident handling
        Security Operations | soc, security operations center
        Threat Intelligence | threat analysis, threat detection, threat modeling, threat hunting
        Encryption | cryptography, pki, public key infrastructure, tls, ssl
        Compliance | regulatory compliance, security compliance, nist, iso 27001, soc 2, hipaa, pci dss, pci, fedramp, gdpr
        Risk Assessment | risk management, it risk management, security risk assessment
        Endpoint Security | endpoint protection, edr, antivirus
        Zero Trust |
        Security Clearance | secret clearance, top secret clearance, ts/sci, ts/sci clearance, active secret clearance
        CompTIA Security+ | security+, security plus, comptia security plus
        CISSP |
    """,
    "Networking & IT": """
        Networking | computer networking, network administration, network engineering, network management, networks, network design, network architecture
        TCP/IP | tcp, tcp ip, ip networking
        DNS | dns management
        DHCP |
        LAN/WAN | lan, wan, lan/wan, wide area network, local area network
        Routing and Switching | routing, switching, routers, switches, bgp, ospf, eigrp
        Cisco | cisco networking, cisco ios, ccna, ccnp
        VPN | vpns, virtual private network
        Wireless Networking | wifi, wi-fi, wlan, wireless
        Network Troubleshooting |
        Load Balancing | load balancers, f5
        Active Directory | ad, microsoft active directory, azure active directory, azure ad, entra id, ldap, group policy
        Microsoft 365 | office 365, o365, m365, microsoft office 365
        Microsoft Exchange | exchange, exchange server, exchange online
        Microsoft SharePoint | sharepoint
        Microsoft Intune | intune
        SCCM | microsoft sccm, mecm, endpoint configuration manager
        VMware | vmware vsphere, vsphere, esxi, vcenter
        Virtualization | virtual machines, hyper-v, hypervisor
        Windows Server | windows server administration, microsoft windows server
        Technical Support | it support, desktop support, help desk, helpdesk, service desk, end user support, tier 2 support
        Troubleshooting | technical troubleshooting, problem resolution, root cause analysis
        IT Service Management | itsm, itil, servicenow, incident management, change management, problem management
        Hardware | computer hardware, hardware troubleshooting, hardware installation
        System Administration | systems administration, sysadmin, server administration, server management
        Backup and Recovery | disaster recovery, backup, data backup, business continuity
        Storage | san, nas, storage management, storage area network
        CompTIA A+ | a+, comptia a plus
        VoIP | unified communications, telephony
    """,
    "Testing & QA": """
        Software Testing | testing, qa testing, manual testing, functional testing, test cases, test planning, test plans
        Test Automation | automated testing, automation testing, test automation frameworks
        Quality Assurance | qa, quality control
        Unit Testing | unit tests, junit, pytest, jest, nunit, testng, mocha
        Integration Testing | integration tests, end-to-end testing, e2e testing, system testing, regression testing
        Selenium | selenium webdriver
        Cypress |
        Playwright |
        Performance Testing | load testing, stress testing, jmeter, loadrunner
        API Testing | postman, soapui, rest assured
        Test-Driven Development | tdd, test driven development
        Behavior-Driven Development | bdd, cucumber, gherkin
        Mobile Testing | appium
        User Acceptance Testing | uat
        Debugging | troubleshooting code, defect tracking, bug tracking
    """,
    "Operating Systems": """
        Linux | linux os, red hat, red hat enterprise linux, rhel, ubuntu, centos, debian
        Unix | unix systems, solaris, aix
        Windows | microsoft windows, windows os, windows 10, windows 11
        macOS | mac os, apple macos, os x
        Embedded Systems | embedded software, embedded, firmware, rtos, embedded c
        Operating Systems | os, operating system
    """,
    "Developer Tools": """
        Git | git version control, gitflow
        GitHub | github enterprise
        GitLab |
        Bitbucket |
        Version Control | version control systems, source control, svn, subversion
        Jira | atlassian jira, jira software
        Confluence | atlassian confluence
        Visual Studio | microsoft visual studio
        VS Code | visual studio code, vscode
        IntelliJ IDEA | intellij
        Eclipse |
        Maven | apache maven
        Gradle |
        npm | yarn
        Linux Command Line | command line, cli, terminal
        XML | extensible markup language, extensible markup language xml
        JSON |
        YAML |
    """,
    "Practices & Methods": """
        Agile | agile methodologies, agile methodology, agile development, agile software development, agile practices, kanban
        Scrum | scrum methodology, scrum master, sprint planning
        Waterfall |
        Software Development Life Cycle | sdlc, software development lifecycle, software development life cycle sdlc
        Software Development | software engineering, application development, programming, coding, software design
        Code Review | code reviews, peer code review
        Technical Documentation | documentation, technical writing, software documentation
        Requirements Gathering | requirements analysis, business requirements, requirements management, requirements
        Systems Engineering | systems integration, system integration
        Performance Optimization | performance tuning, optimization, query optimization
        Project Management | it project management, technical project management, project planning, pmp
        Product Management | product development, product strategy, product roadmap
        Program Management | technical program management, tpm
        Business Analysis | business analyst, business process analysis, process improvement, process mapping
        DevSecOps |
        Open Source | open source software, open-source
        UI/UX Design | ux design, ui design, user experience, user interface design, ux, ui, figma, wireframing, prototyping
    """,
    "Enterprise Software": """
        Salesforce | salesforce crm, salesforce.com, salesforce software, apex, salesforce administration
        SAP | sap software, sap erp, sap hana, s/4hana, abap
        Oracle E-Business Suite | oracle ebs, oracle applications, oracle peoplesoft, peoplesoft
        Workday |
        ServiceNow Platform | servicenow development, servicenow administration
        Dynamics 365 | microsoft dynamics, microsoft dynamics 365, dynamics crm
        ERP Systems | erp, enterprise resource planning
        CRM Systems | crm, customer relationship management
        Microsoft Power Platform | power apps, powerapps, power automate, microsoft power automate
        Microsoft Office | microsoft office suite, ms office, office suite, microsoft word, microsoft powerpoint, microsoft outlook
    """,
    "Soft Skills": """
        Communication | communication skills, verbal communication, written communication, oral communication, written and verbal communication, excellent communication skills, interpersonal communication
        Problem Solving | problemsolving, problem-solving, problem solving skills, problemsolving skills, critical thinking, analytical thinking
        Analytical Skills | analytical, analytical abilities, analysis
        Teamwork | collaboration, team player, teamwork and collaboration, team collaboration, cross-functional collaboration, cross-functional teams
        Decision Making | decision-making, sound judgment
        Leadership | team leadership, technical leadership, leadership skills, people management, team management
        Mentoring | coaching, mentorship
        Attention to Detail | detail oriented, detail-oriented
        Time Management | prioritization, multitasking, organizational skills, organization
        Interpersonal Skills | interpersonal, relationship building
        Customer Service | customer support, client service, customer focus
        Adaptability | flexibility, adaptable, learning agility
        Stakeholder Management | stakeholder communication, stakeholder engagement, client relationship management
        Presentation Skills | presentations, public speaking
        Self-Motivation | self-motivated, self starter, initiative, independent work, ability to work independently
        Training | training and development, user training, end user training
    """,
}
