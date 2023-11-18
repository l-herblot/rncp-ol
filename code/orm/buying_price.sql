create table if not exists public.buying_price_market_type
(
    id serial primary key,
    market_type_name character varying(255) collate pg_catalog."default",
    key  character varying(255) collate pg_catalog."default",
    active boolean
);
insert into public.buying_price_market_type (id, market_type_name, key, active) values
(1, 'Fruits et Légumes', 'FRUITS-ET-LEGUMES', true),
(2, 'Fleurs & plantes ornementale', 'FLEURS---PLANTES-ORNEMENTALES', false),
(3, 'Pêche et aquaculture', 'PECHE-ET-AQUACULTURE', true),
(4, 'Beurre Oeuf Fromage', 'BEURRE-OEUF-FROMAGE', true),
(5, 'Viande', 'VIANDE', true);

create table if not exists public.buying_price_market_step
(
    id serial primary key,
    market_step_name character varying(255) collate pg_catalog."default",
    key  character varying(255) collate pg_catalog."default",
    active boolean
);
insert into public.buying_price_market_step (id, market_step_name, key, active) values
(1, 'Production', 'PRODUCTION', false),
(2, 'Marché de producteurs', 'MARCHE-DE-PRODUCTEURS', false),
(3, 'Expédition', 'EXPEDITION', false),
(4, 'Import', 'IMPORT', false),
(5, 'Grossistes', 'GROSSISTES', true),
(6, 'Détail', 'DETAIL', false);

create table if not exists public.buying_price_market_step_type
(
    id serial primary key,
    id_market_step integer references  public.buying_price_market_step (id) on delete cascade,
    id_market_type integer references  public.buying_price_market_type (id) on delete cascade,
    active boolean
);
insert into public.buying_price_market_step_type (id_market_step, id_market_type, active) values
(1,1, false), (2,1, false), (3,1, false), (4,1, false), (5,1, true), (6,1, false),
(1,2, false), (3,2, false), (5,2, false),
(5,3, true), (6,3, false),
(5,4, true), (6,4, false),
(1,5, false), (3,5, false), (5,5, true), (6,5, false),;

create table if not exists public.buying_price_market
(
    id serial primary key,
    id_market_step integer references  public.buying_price_market_step (id) on delete cascade,
    id_market_type integer references  public.buying_price_market_type (id) on delete cascade,
    market_name character varying(255) collate pg_catalog."default",
    key  character varying(255) collate pg_catalog."default",
    active boolean
);
create table if not exists public.buying_price_market_date
(
    id serial primary key,
    id_market integer references  public.buying_price_market (id) on delete cascade,
    market_date  date,
    scrapped boolean

);
create table if not exists public.buying_price_product
(
    id serial primary key,
    id_market integer references public.buying_price_market (id) on delete cascade,
    id_market_type integer references public.buying_price_market_type (id) on delete cascade,
    id_market_step integer references public.buying_price_market_step (id) on delete cascade,
    date_price date,
    product_name character varying(255) collate pg_catalog."default",
    avg_price numeric(10, 3),
    min_price numeric(10, 3),
    max_price numeric(10, 3),
    unit character varying(255) collate pg_catalog."default",
    active boolean
);
create unique index buying_price_product_name ON public.buying_price_product (product_name, id_market, date_price);