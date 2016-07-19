These files contain the original data sent by the LIGO team with an additional column on the end (column 23) which has the first publication time as reported in Hydra in seconds after the published origin time.
The first publication time was taken as the minimum of the following fields: 

TFIRSTEXTP: Time of SINSTFIR (initial external agency publication) (seconds after Origin time) 
TFIRSTATWCINT: Time of first ATWC PUBLIC publication. (seconds after Origin time)  
TFIRSTPTWCINT: Time of first PTWC PUBLIC publication. (seconds after Origin time)
TDETECTLATENCY: This is the detection latency for the hydra automatic system in seconds (i.e. tEventCreatedInHydra - tOrigin). This is the time that Hydra became aware of the event, regardless of the external source of the detection(GLASS, TWC, first-time-caller-long-time-listener, CIA)
TFIRSTPUB: The time (seconds after Origin time) that this event was first publicly released. (Publicly released means released to the world outside of Hydra, which under the current configuration may mean that it was only released to Chixilub.)

