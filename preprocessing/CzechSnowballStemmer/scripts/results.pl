#!/usr/bin/perl
use locale;

# Pocita, kolik vysledku stemmovani bylo spravnych/castecne spravnych/chybnych.
#
# vstup: <spravne vysledky> <ziskane vysledky>
#        print - nepovinne. zpusobi vypsani castecnych a spatnych vysledku
# vystup: statistika spravnosti vysledku
#
# Soubor s vysledky musi mit nasledujici strukturu:
# zadani <nekolik mezer> vysledek

my $progname = $0;

#skript ocekava minimalne 2 parametry: spravne vysledky a ziskane vysledky
if (scalar @ARGV < 2) {
  print "Usage: $progname <file zadani> <file vysledek> [print]\n";
  exit 1;
}
$fileZadani = shift(@ARGV);
$fileVysledek = shift(@ARGV);
$printResult = shift(@ARGV);

@zadani = ();
@vysledky = ();
$line = "";
$i = 0;               #pocet slov ve vysledcich
@dobre = ();          #pole pro ukladani spravnych vysledku
@budiz = ();          #...castecnych...
@spatne = ();         #...spatnych...
$odebranoAkorat = 0;  #pocty over- a under-stemmingu
$odebranoMoc = 0;
$odebranoMalo = 0;

#nacteni zadani
open(FILEZADANI, $fileZadani) or die "Nelze otevrit soubor ".$fileZadani;
while($line = <FILEZADANI>) {
  $line =~ s/\s+$//;
  next if $line eq "";
  push @zadani, $line;
}
close FILEZADANI;

#nacteni vysledky
open(FILEVYSLEDEK, $fileVysledek) or die "Nelze otevrit soubor ".$fileVysledek;
while($line = <FILEVYSLEDEK>) {
  $line =~ s/\s+$//;
  next if $line eq "";
  push @vysledky, $line;
}
close FILEVYSLEDEK;


foreach $res (@vysledky) {
  #pokud by Snowball vkladal misto smazane koncove samohlasky znak X, bude to treba korigovat
  $korekceSamohlasek = 0; 
  # $1 = zadani
  # $2 = mezery mezi zadanim a vysledkem
  # $3 = vysledek
  $res =~ /(.*?)(\s+)([a-záčďéěíňóřšťúůýž].*)/;
  $vysledek = $3;

  if($vysledek =~ /X$/) {   #pokud vysledek konci na X
    $korekceSamohlasek = 1; #doslo v algoritmu k nahrazeni koncove samohlasky za X
    $vysledek =~ s/X$//g;   #odstraneni znaku X
  }
  
  #format souboru se zadanim je:
  #zadani/spravny vysledek/castecny vysledek/
  $zadani[$i] =~ /^(.*?)\/(.*?)\/(.*?)$/; #rozdeleni zaznamu podle znaku /
  $zadaneSlovo = $1;
  $spravnyVysl = $2;
  $budizVysl = $3;


  $report = $zadaneSlovo." / ".$spravnyVysl." / ".$budizVysl."          -> ".$vysledek;
  if($korekceSamohlasek eq 1) {$report = $report."X";}
  
  if($vysledek eq $spravnyVysl) {push @dobre, $report;}   #do pole @dobre pridan vysledek
  elsif($vysledek eq $budizVysl) {push @budiz, $report;}  #castecne spravny vysledek
  else { #spatny vysledek
    push @spatne, $report;
    #zjisteni, jestli se jedna o over- nebo under-stemming
    if(length($vysledek) gt length($spravnyVysl)) {$odebranoMalo++;}
    elsif(length($vysledek) lt length($spravnyVysl)) {$odebranoMoc++;}
    else {$odebranoAkorat++;}
  }
  
  $i++; #pocet slov ve vysledcich
}


$dobre = @dobre;    #pocet spravnych
$budiz = @budiz;    #pocet castecnych
$spatne = @spatne;  #pocet spatnych
#vypocty procent
$procSpatnych = ($spatne * 100) / $i;
$procDobBud = (($dobre+$budiz)*100) / $i;
$procDobrych = ($dobre * 100) / $i;
$procBudiz = ($budiz * 100) / $i;


printf("%-10s % 5d        (%.1f%% = %.1f%% + %.1f%%)\n", "dobře:", $dobre, $procDobBud, $procDobrych, $procBudiz);
printf("%-10s % 5d \n", "částečně:", $budiz);
printf("%-10s % 5d z %d (%.1f%%)\n", "špatně:", $spatne, $i, $procSpatnych);
printf("%-10s % 5d/%d/%d  málo/moc/správně\n", "chyba:", $odebranoMalo, $odebranoMoc, $odebranoAkorat);


#s parametrem "print" se tisknou castecne a spatne vysledky
if($printResult eq "print") {
  print STDOUT "-----------------------------------\nČÁSTEČNĚ\n-----------------------------------\n";
  foreach $res (@budiz) {
    print STDOUT $res."\n";
  }
  print STDOUT "-----------------------------------\nŠPATNĚ\n-----------------------------------\n";
  foreach $res (@spatne) {
    print STDOUT $res."\n";
  }
}