-- Tabela principal de processos
CREATE TABLE IF NOT EXISTS processos (
    id SERIAL PRIMARY KEY,
    numero_processo VARCHAR(50) UNIQUE NOT NULL,
    url_original TEXT NOT NULL,
    html_completo TEXT,
    data_coleta TIMESTAMP DEFAULT NOW(),
    status_processamento VARCHAR(20) DEFAULT 'pendente',
    hash_documento VARCHAR(64),
    tentativas INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Tabela de metadados
CREATE TABLE IF NOT EXISTS processos_metadados (
    id SERIAL PRIMARY KEY,
    processo_id INTEGER REFERENCES processos(id) ON DELETE CASCADE,
    orgao_julgador TEXT,
    orgao_julgador_colegiado TEXT,
    relator TEXT,
    classe_judicial TEXT,
    competencia TEXT,
    assunto_principal TEXT,
    autor TEXT,
    reu TEXT,
    data_publicacao DATE,
    tipo_decisao VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Tabela de conteúdo
CREATE TABLE IF NOT EXISTS processos_conteudo (
    id SERIAL PRIMARY KEY,
    processo_id INTEGER REFERENCES processos(id) ON DELETE CASCADE,
    tipo_secao VARCHAR(50),
    conteudo_html TEXT,
    conteudo_texto TEXT,
    conteudo_limpo TEXT,
    ordem INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Tabela de log
CREATE TABLE IF NOT EXISTS scraping_log (
    id SERIAL PRIMARY KEY,
    url TEXT,
    tipo_operacao VARCHAR(20),
    status VARCHAR(20),
    tentativas INTEGER DEFAULT 0,
    erro_mensagem TEXT,
    timestamp_inicio TIMESTAMP,
    timestamp_fim TIMESTAMP,
    duracao_segundos NUMERIC(10,2)
);

-- Índices
CREATE INDEX IF NOT EXISTS idx_processo_numero ON processos(numero_processo);
CREATE INDEX IF NOT EXISTS idx_processo_status ON processos(status_processamento);
CREATE INDEX IF NOT EXISTS idx_metadados_processo ON processos_metadados(processo_id);
CREATE INDEX IF NOT EXISTS idx_conteudo_processo ON processos_conteudo(processo_id);
