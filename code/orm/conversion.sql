create table if not exists public.ingredients_conversion (
    id serial primary key,
    ingredient_name varchar(255) collate pg_catalog."default",
    unit varchar(255) collate pg_catalog."default",
    weight numeric(10, 3),
    language varchar(4) collate pg_catalog."default"
);
insert into ingredients_conversion (ingredient_name,unit,weight,language) values
('Aïl','tete',80,'fr'),
('Aïl','gousse',7,'fr'),
('Aïl','cas',9,'fr'),
('Aïl','cac',3,'fr');



