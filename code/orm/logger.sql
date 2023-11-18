CREATE TABLE IF NOT EXISTS public.logging
(
	id serial4 NOT NULL,
	event_time timestamp NOT NULL,
	event_type varchar(20) NULL,
	logger_id text NULL,
	trigger text NULL,
	message text NULL,
	CONSTRAINT logging_pk PRIMARY KEY (id)
);
CREATE INDEX IF NOT EXISTS logging_event_type_idx ON public.logging USING btree (event_type);

CREATE TABLE IF NOT EXISTS public.logging_queue
(
	id serial4 NOT NULL,
	event_time timestamp NULL,
	event_type varchar(20) NULL,
	logger_id text NULL,
	trigger text NULL,
	message text NULL,
	CONSTRAINT logging_tracking_pk PRIMARY KEY (id)
);
